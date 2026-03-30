# run/entry_fedlab_native.py
import argparse, os, yaml, random, torch
from typing import List
from server.fedavg_native_handler import FedAvgNativeServerHandler
from client.NativeClientTrainer import NativeClientTrainer
from factories.model_factory import build_model_and_processor
from factories.data_factory import build_central_eval_loaders
from fedlab.utils import SerializationTool
import sys, os, time


def load_yaml(path): 
    with open(path, "r") as f: 
        return yaml.safe_load(f)

def _coerce_train_types(cfg):
    t = cfg.setdefault("train", {})
    for k in ["lr", "wd"]:
        if k in t: t[k] = float(t[k])
    for k in ["batch_size", "local_epochs", "max_steps", "eval_every"]:
        if k in t: t[k] = int(t[k])
    r = cfg.setdefault("runtime", {})
    for k in ["rounds", "num_clients"]:
        if k in r: r[k] = int(r[k])
    if "sample_ratio" in r: r["sample_ratio"] = float(r["sample_ratio"])
    return cfg

@torch.no_grad()
def eval_central(cfg, server_handler, proc_or_tok, split_loader):
    if split_loader is None:
        return {"accuracy": None}
    # 构建同构评估模型，并反序列化当前全局参数
    model, _ = build_model_and_processor(cfg)
    SerializationTool.deserialize_model(model, server_handler.model_parameters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    correct, total = 0, 0
    for batch in split_loader:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits if hasattr(out, "logits") else out[0]
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += (preds == labels).sum().item()
        total   += labels.numel()
    return {"accuracy": correct / max(total, 1)}

def build_client(cid):
    return NativeClientTrainer(cfg, cid)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"fedlab_run_{timestamp}.log")
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout
    print(f"[LOGGING] Log file: {log_path}\n")
    return log_path

def main():
    # === 日志文件设置 ===
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"fedlab_run_{timestamp}.log")

    # 打开日志文件并重定向 stdout/stderr
    log_file = open(log_path, "a")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"[LOGGING] All print outputs are being written to {log_path}\n")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_task.yaml")
    args = ap.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)
    cfg = _coerce_train_types(load_yaml(args.config))

    # —— 构建全局模型（与客户端同构） —— #
    global_model, proc_or_tok = build_model_and_processor(cfg)
    global_model.to("cpu")
    server = FedAvgNativeServerHandler(
        model=global_model,
        global_round=cfg["runtime"]["rounds"],
        sample_ratio=cfg["runtime"]["sample_ratio"],
        cuda=False, logger=None
    )
    server.client_num_in_total = cfg["runtime"]["num_clients"]

    # —— 集中式评估数据（不切片） —— #
    central_val, central_test = build_central_eval_loaders(cfg, proc_or_tok)

    # —— 保存目录 & 评估控制 —— #
    save_root = os.path.join(cfg.get("io", {}).get("save_dir", "checkpoints"), "global")
    os.makedirs(save_root, exist_ok=True)
    best_global = -1.0
    best_path = os.path.join(save_root, "best.pt")
    eval_every = int(cfg.get("eval", {}).get("every", 1))

    # —— 按需构建客户端的工厂 —— #
    def build_client(cid, _cfg=cfg):
        return NativeClientTrainer(_cfg, cid)

    prev_round = server.round
    while not server.if_stop:
        # 1) 采样
        selected = server.sample_clients()
        # 2) 下发
        payload = server.downlink_package  
        # 3) 客户端本地训练→上传
        for cid in selected:
            client = build_client(cid)
            serialized_client, metrics, improved = client.local_process(payload, round_idx=server.round)
            client.offload_and_cleanup()  # 释放显存
            server._update_global_model([serialized_client])  

        # 4) 若本轮完成（round 发生变化），做集中式评估与保存
        if server.round != prev_round:
            r = server.round
            print(f"[Round {r}] aggregated.")
            if r % eval_every == 0:

                val_metrics = eval_central(cfg, server, proc_or_tok, central_val)
                print(f"[Central VAL] round={r} acc={val_metrics['accuracy']}")
                snap_path = os.path.join(save_root, f"round_{r:03d}.pt")
                torch.save(global_model.state_dict(), snap_path)
                if val_metrics["accuracy"] is not None and val_metrics["accuracy"] > best_global:
                    best_global = val_metrics["accuracy"]
                    torch.save(global_model.state_dict(), best_path)
            prev_round = server.round

    # 5) 训练结束：集中式 TEST
    test_metrics = eval_central(cfg, server, proc_or_tok, central_test)
    print(f"[Central TEST] acc={test_metrics['accuracy']}")

    # 6) 客户端个性化 TEST：加载各自 best.pt 评估
    for cid in range(cfg["runtime"]["num_clients"]):
        c = build_client(cid) 
        best_path_client = c.best_path if os.path.exists(c.best_path) \
                           else os.path.join(c.save_root, f"round_{server.round:03d}.pt")
        if os.path.exists(best_path_client):
            state = torch.load(best_path_client, map_location="cpu")
            c.model.load_state_dict(state)
        cm = c.evaluate(split="test")
        print(f"[Client {cid:03d} TEST] acc={cm['accuracy']}")
        c.offload_and_cleanup()

if __name__ == "__main__":
    main()
