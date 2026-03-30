

import os
import sys
import yaml
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any
import time

import torch
from fedlab.utils import SerializationTool
import torch.multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from factories.model_factory import build_model_and_processor
from client.NativeClientTrainer import NativeClientTrainer
from server.mmoe_meta_handler import GroupMMOEHandler
from data_specs.multidomain_indexer import build_and_save_specs, load_specs


# ---------------------- 工具函数 ----------------------
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _coerce(cfg: dict) -> dict:

    t = cfg.setdefault("train", {})
    for k in ["lr", "wd"]:
        if k in t:
            t[k] = float(t[k])
    for k in ["batch_size", "local_epochs", "max_steps", "eval_every", "grad_accum_steps"]:
        if k in t:
            t[k] = int(t[k])
    r = cfg.setdefault("runtime", {})
    for k in ["rounds", "num_clients"]:
        if k in r:
            r[k] = int(r[k])
    if "sample_ratio" in r:
        r["sample_ratio"] = float(r["sample_ratio"])

    m = cfg.setdefault("meta", {})
    m.setdefault("K", 2)          
    m.setdefault("tau", 1.0)      
    m.setdefault("eta", 0.5)      
    m.setdefault("epsilon", 1e-6) 
    m.setdefault("inner_steps", 1)        
    m.setdefault("proxy_batches", 1)      
    m.setdefault("meta_inner_lr", 1e-3)   
    return cfg

@torch.no_grad()
def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return SerializationTool.serialize_model(model).view(-1).detach().cpu()

def inner_adapt_on_proxy(cfg: dict, base_vec: torch.Tensor, proxy_batches: List[Dict]) -> torch.Tensor:
    model, _ = build_model_and_processor(cfg)
    SerializationTool.deserialize_model(model, base_vec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    lr_inner = float(cfg["meta"].get("meta_inner_lr", 1e-3))
    steps = int(cfg["meta"].get("inner_steps", 1))
    optim = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_inner, momentum=0.0
    )

    for _ in range(steps):
        for batch in proxy_batches:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out
            optim.zero_grad(); loss.backward(); optim.step()

    theta_star = flatten_params(model)
    delta = theta_star - base_vec

    model.to("cpu"); torch.cuda.empty_cache()
    return delta


# ---------------------- 多进程客户端 worker ----------------------
def _client_worker(cfg_base: dict, spec, payload: List[torch.Tensor], round_idx: int, gpu_id: int, q):

    import copy
    try:
        cfg = copy.deepcopy(cfg_base)
        cfg.setdefault("runtime", {})["gpu_id"] = int(gpu_id)

        trainer = NativeClientTrainer(cfg, spec)
        serialized, metrics, improved = trainer.local_process(payload, round_idx=round_idx)
        q.put((int(spec.cid), serialized.cpu(), metrics, bool(improved)))
    except Exception as e:
        q.put((int(spec.cid), None, {"error": repr(e)}, False))
    finally:
        try:
            if 'trainer' in locals() and hasattr(trainer, "offload_and_cleanup"):
                trainer.offload_and_cleanup()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def build_flat_index_mask_for_lora(model) -> np.ndarray:

    name_and_shapes = []
    total = 0
    for n, p in model.named_parameters():
        numel = p.numel()
        name_and_shapes.append((n, numel, p.requires_grad))
        total += numel

    mask = np.zeros(total, dtype=np.bool_)
    has_lora = False

    lora_keys = ("lora", "lora_A", "lora_B", "adapter")  
    cursor = 0
    for n, numel, req in name_and_shapes:
        hit = any(k in n.lower() for k in lora_keys)
        if hit:
            mask[cursor:cursor+numel] = True
            has_lora = True
        cursor += numel

    if not has_lora:

        cursor = 0
        for n, numel, req in name_and_shapes:
            if req:
                mask[cursor:cursor+numel] = True
            cursor += numel

    return mask


def compute_and_dump_similarity_csv(latest_vecs, lora_mask: np.ndarray, out_csv: str):

    n = len(latest_vecs)
    if n == 0:
        return
    dim = int(lora_mask.sum())
    if dim == 0:
        print("[Sim] lora_mask 为空，跳过相似度计算")
        return

    X = np.zeros((n, dim), dtype=np.float32)
    has_vec = np.zeros(n, dtype=np.bool_)
    for i, v in enumerate(latest_vecs):
        if v is None:
            continue
        vv = v.view(-1).numpy()
        if vv.shape[0] < lora_mask.shape[0]:
            use_len = vv.shape[0]
            submask = lora_mask[:use_len]
            X[i] = vv[:use_len][submask]
        else:
            X[i] = vv[lora_mask]
        has_vec[i] = True

    norms = np.linalg.norm(X, axis=1, keepdims=True)  # [n,1]
    norms[norms == 0] = 1.0
    Xn = X / norms

    S = np.matmul(Xn, Xn.T)  
    for i in range(n):
        if not has_vec[i]:
            S[i, :] = 0.0
            S[:, i] = 0.0

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, S, delimiter=",")
    print(f"[Sim] LoRA cosine similarity matrix saved to: {out_csv} (shape={S.shape})")


def _ensure_metrics_csv(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_round_client_metrics.csv")
    if not os.path.exists(path):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "cid", "group",
                "loss_evals", "loss_sum", "loss_mean",
                "val_acc", "val_loss", "test_acc", "test_loss",
                "loss_trace_path"
            ])
    return path

def _append_metrics_row(csv_path: str, round_idx: int, cid: int, group: int, metrics: Dict[str, Any]):
    import csv
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            int(round_idx), int(cid), int(group),
            int(metrics.get("loss_evals", -1)),
            float(metrics.get("loss_sum", float("nan"))),
            float(metrics.get("loss_mean", float("nan"))),
            float(metrics.get("val_acc", float("nan"))),
            float(metrics.get("val_loss", float("nan"))),
            float(metrics.get("test_acc", float("nan"))),
            float(metrics.get("test_loss", float("nan"))),
            metrics.get("loss_trace_path", ""),
        ])


# ---------------------- 主流程 ----------------------
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_task.yaml")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)

    cfg = _coerce(load_yaml(args.config))

    # ====== 加载/生成 ClientSpec======
    spec_path = cfg["federation"]["save_spec_path"]
    if os.path.exists(spec_path):
        specs = load_specs(spec_path)
    else:
        specs = build_and_save_specs(cfg)  

    m = len(specs)
    if m == 0:
        raise RuntimeError(
            f"[SpecError] No clients were created (m=0). "
            f"Check dataset.root / dataset.categories / amazon.domains.\n"
            f"Spec path: {spec_path}"
        )

    cats = sorted({s.category for s in specs})
    cat2id = {c: i for i, c in enumerate(cats)}
    for s in specs:
        if not hasattr(s, "category_id") or s.category_id is None:
            s.category_id = cat2id[s.category]
    num_groups = len(cats)
    print(f"[Spec] clients={m}, groups={num_groups}, categories={cats}")

    # ==========  MMOE Handler ==========
    global_model, _ = build_model_and_processor(cfg)
    lora_mask = build_flat_index_mask_for_lora(global_model)
    global_model.to("cpu") 
    h = GroupMMOEHandler(
        model=global_model,
        global_round=cfg["runtime"]["rounds"],
        sample_ratio=cfg["runtime"]["sample_ratio"],
        K=num_groups,                            
        tau=cfg["meta"]["tau"],
        eta=cfg["meta"]["eta"],
        epsilon=cfg["meta"]["epsilon"],
        inner_steps=cfg["meta"]["inner_steps"],
        cuda=False, logger=None
    )
    h.client_num_in_total = m

    h.group_of = [s.category_id for s in specs]
    h._group_logits = torch.zeros(h.K, m)
    h._alpha = torch.full((h.K, m), 1.0 / m)

    latest_vecs: List[torch.Tensor] = [None] * m

    def build_client(idx: int) -> NativeClientTrainer:
        return NativeClientTrainer(cfg, specs[idx])


    if h._theta is None:
        base = h.model_parameters.view(-1).detach().cpu()
        h._theta = [base.clone() for _ in range(h.K)]


    gpu_list: List[int] = list(map(int, cfg.get("runtime", {}).get("gpus", [0])))
    clients_per_gpu = int(cfg.get("runtime", {}).get("clients_per_gpu", 1))
    max_workers = max(1, len(gpu_list) * clients_per_gpu)

    out_dir = cfg.get("runtime", {}).get("out_dir", "./outputs")
    metrics_csv_path = _ensure_metrics_csv(out_dir)

    prev_round = h.round
    print(f"[Init] MMOE meta runner start. total_clients={m}, groups={h.K}")

    while not h.if_stop:
        selected = h.sample_clients()
        print(f"[Dispatch] round={h.round} selected={selected}")

        # =============== 客户端阶段：并行执行本地训练 ===============
        if max_workers <= 1 or len(selected) == 1:
            for cid in selected:
                payload = h.broadcast_for(cid)
                client = build_client(cid)
                serialized_client, metrics, improved = client.local_process(payload, round_idx=h.round)

                latest_vecs[cid] = serialized_client.view(-1).detach().cpu()
                h._update_global_model([serialized_client])

                _append_metrics_row(metrics_csv_path, h.round, cid, h.group_of[cid], metrics)

                if hasattr(client, "offload_and_cleanup"):
                    client.offload_and_cleanup()
                del client
        else:

            for start in range(0, len(selected), max_workers):
                batch_ids = selected[start:start + max_workers]
                q = mp.Queue()
                procs = []

                for i, cid in enumerate(batch_ids):
                    gpu_id = gpu_list[i % len(gpu_list)]
                    payload = h.broadcast_for(cid)
                    p = mp.Process(
                        target=_client_worker,
                        args=(cfg, specs[cid], payload, h.round, gpu_id, q)
                    )
                    p.start()
                    procs.append(p)

                results: List[Tuple[int, torch.Tensor, Dict, bool]] = []
                for _ in range(len(batch_ids)):
                    results.append(q.get())


                for p in procs:
                    p.join()

                for (cid, serialized_client, metrics, improved) in results:
                    if serialized_client is None:
                        print(f"[Client {cid}] Worker error: {metrics.get('error')}")
                        continue
                    latest_vecs[cid] = serialized_client.view(-1).detach().cpu()
                    h._update_global_model([serialized_client])

                    _append_metrics_row(metrics_csv_path, h.round, cid, h.group_of[cid], metrics)

        # ===============MMOE 元聚合 ===============
        start = time.time()
        if h.round != prev_round:
            r = h.round
            print(f"[Aggregate] round={r} fedlab-count reached, start MMOE meta-update")

            missing = sum(1 for v in latest_vecs if v is None)
            if missing > 0:
                print(f"[Warn] {missing}/{m} clients have no upload yet; using placeholders.")
            if h._theta is None:
                base_vec = h.model_parameters.view(-1).detach().cpu()
                placeholders = [base_vec for _ in range(m)]
            else:
                theta_mean = torch.stack(h._theta, dim=1).mean(dim=1)  # [D]
                placeholders = [theta_mean for _ in range(m)]

            cols = []
            for i in range(m):
                cols.append(latest_vecs[i] if latest_vecs[i] is not None else placeholders[i].clone())
            V_all = torch.stack(cols, dim=1)  # [D, m]
            V_all = V_all + 1e-8 * torch.randn_like(V_all)

            csv_path = os.path.join(out_dir, f"lora_similarity_round{r}.csv")
            compute_and_dump_similarity_csv(latest_vecs, lora_mask, csv_path)
            h._compute_group_thetas(V_all, m)
            proxy_deltas = [None for _ in range(h.K)]
            for k in range(h.K):
                gid_indices = [i for i, g in enumerate(h.group_of) if g == k]
                if not gid_indices:
                    continue
                cidx = gid_indices[0]
                client = build_client(cidx)
                proxy_batches = client.sample_proxy_batches(max_batches=int(cfg["meta"]["proxy_batches"]))
                base_theta_k = h._theta[k]
                delta_theta = inner_adapt_on_proxy(cfg, base_theta_k, proxy_batches)
                proxy_deltas[k] = delta_theta
                if hasattr(client, "offload_and_cleanup"):
                    client.offload_and_cleanup()
                del client

            
            h.meta_update(V_all, proxy_deltas)

            print(f"[Meta] round={r} alpha-norms={[float(a.norm().item()) for a in h._alpha]}")
            prev_round = h.round

            
            with torch.no_grad():
                alpha = torch.softmax(h._group_logits / h.tau, dim=1)  # [K, m]
                for k in range(h.K):
                    ak = alpha[k]                 # [m]
                    vals, idx = torch.topk(ak, k=min(5, ak.numel()))
                    print(f"[Meta] group={k} top5 clients: "
                          f"{[(int(i), float(v)) for i, v in zip(idx, vals)]}  sum={float(ak.sum())}")
        end = time.time()
        agg_time_ms = (end - start) * 1000
        print(f"[Aggregation Time] {agg_time_ms:.3f} ms")


    print("[Finish] MMOE meta-aggregation training done.")


if __name__ == "__main__":
    main()
