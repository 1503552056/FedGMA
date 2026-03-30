
from typing import List, Dict, Any
import os
import csv
import time

import torch
from torch.amp import GradScaler, autocast

from factories.data_factory import build_dataloaders_from_spec
from factories.model_factory import build_model_and_processor
from fedlab.utils import SerializationTool


class NativeClientTrainer:
    def __init__(self, cfg: Dict[str, Any], client_spec):
        self.cfg = cfg
        self.spec = client_spec

        gpu_id = int(cfg.get("runtime", {}).get("gpu_id", 0))
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_id)
                self.device = torch.device(f"cuda:{gpu_id}")
            except Exception:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model, self.proc = build_model_and_processor(cfg)
        self.model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders_from_spec(
            cfg, self.spec, self.proc
        )
        self.sample_loader = None
        dataset_name = str(getattr(self.spec, "dataset", "")).lower()

        if dataset_name == "pkl_partition":
            try:
                root = cfg["dataset"]["root"]
                cid = int(self.spec.cid)
                sample_pkl = os.path.join(root, "sample", f"client{cid}.pkl")
                if os.path.exists(sample_pkl):
                    ds_sample = torch.load(sample_pkl, weights_only=False)
                    from torch.utils.data import DataLoader
                    batch_size = int(cfg["train"]["batch_size"])
                    self.sample_loader = DataLoader(
                        ds_sample,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                    )
                    print(f"[Client {cid}] Loaded sample dataset from {sample_pkl}")
                else:
                    print(f"[Client {cid}] sample pkl not found, fallback to train_loader.")
            except Exception as e:
                print(f"[Client {self.spec.cid}] failed to build sample_loader: {e}")
                self.sample_loader = None


        self.epochs = int(cfg["train"]["local_epochs"])
        lr = float(cfg["train"]["lr"])
        wd = float(cfg["train"].get("wd", 0.0))
        self.grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
        self.use_amp = bool(cfg["train"].get("use_amp", False)) and torch.cuda.is_available()
        self.max_steps = int(cfg["train"].get("max_steps", 0))  

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=wd
        )
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)


        self.best_val = -1e18
        self.best_payload = None


        self.out_dir = cfg.get("runtime", {}).get("out_dir", "./outputs")
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_every = int(cfg.get("train", {}).get("log_loss_every", 20)) 

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        out = self.model(**batch)
        return out.loss if hasattr(out, "loss") else out

    def local_process(self, downlink_package: List[torch.Tensor], round_idx: int = 0):

        if downlink_package is not None and len(downlink_package) > 0:
            SerializationTool.deserialize_model(self.model, downlink_package[0])

        print(
            f"[Client {self.spec.cid}] Start local training | "
            f"epochs={self.epochs}, train_size={len(self.train_loader.dataset)}"
        )

        cid = int(self.spec.cid)
        trace_path = os.path.join(self.out_dir, f"loss_trace_round{int(round_idx)}_cid{cid}.csv")
        summary_path = os.path.join(self.out_dir, "loss_summary_per_round.csv")

        with open(trace_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "cid", "local_step", "epoch", "loss"])


        loss_evals = 0
        loss_sum = 0.0
        local_step = 0
        epoch_idx = 0

        self.model.train()
        step = 0
        got_first_batch_logged = False


        dynamic_mbs = None  

        for ep in range(self.epochs):
            print(f"[Client {self.spec.cid}] Epoch {ep} begin")
            for batch_idx, batch in enumerate(self.train_loader):
                if not got_first_batch_logged:
                    print(f"[Client {self.spec.cid}] Got first batch")
                    got_first_batch_logged = True


                bs = int(batch["labels"].size(0)) if "labels" in batch else None
                if bs is None:
                    for v in batch.values():
                        if torch.is_tensor(v) and v.ndim >= 1:
                            bs = int(v.size(0))
                            break
                if bs is None:
                    raise RuntimeError("[NativeClientTrainer] Cannot infer batch size from batch!")

                if dynamic_mbs is None or dynamic_mbs > bs:
                    dynamic_mbs = max(1, bs)

                start = 0
                while start < bs:
                    end = min(start + dynamic_mbs, bs)
                    micro = {}
                    for k, v in batch.items():
                        if torch.is_tensor(v) and v.ndim >= 1 and v.size(0) == bs:
                            micro[k] = v[start:end]
                        else:
                            micro[k] = v 

                    micro = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in micro.items()}

                    try:
                        with autocast(device_type="cuda", enabled=self.use_amp):
 
                            loss = self._forward_loss(micro) / self.grad_accum_steps


                        loss_value = float(loss.detach().item())
                        loss_evals += 1
                        loss_sum += loss_value
                        local_step += 1
                        if self.log_every > 0 and (local_step % self.log_every == 0):
                            with open(trace_path, "a", newline="") as f:
                                csv.writer(f).writerow([int(round_idx), cid, local_step, epoch_idx, loss_value])


                        self.scaler.scale(loss).backward()
                        step += 1

                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if dynamic_mbs > 1:
                            dynamic_mbs = max(1, dynamic_mbs // 2)
                            print(
                                f"[Client {self.spec.cid}] OOM caught → shrink microbatch to {dynamic_mbs} and retry"
                            )
                            continue  
                        else:
                            raise  


                    if (step % self.grad_accum_steps) == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    start = end 


                if self.max_steps > 0 and step >= self.max_steps:
                    print(f"[Client {self.spec.cid}] Hit max_steps={self.max_steps}, break epoch")
                    break


            epoch_mean = (loss_sum / loss_evals) if loss_evals > 0 else 0.0
            with open(trace_path, "a", newline="") as f:
                csv.writer(f).writerow([int(round_idx), cid, -1, epoch_idx, epoch_mean])
            epoch_idx += 1

            if self.max_steps > 0 and step >= self.max_steps:
                break


        serialized = SerializationTool.serialize_model(self.model).detach().cpu()


        val_res = self.evaluate(split="val")
        test_res = self.evaluate(split="test")


        round_mean = (loss_sum / loss_evals) if loss_evals > 0 else 0.0
        need_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(["round", "cid", "loss_evals", "loss_sum", "loss_mean", "timestamp"])
            writer.writerow([int(round_idx), cid, int(loss_evals), float(loss_sum), float(round_mean), int(time.time())])

        improved = False
        val_metric = val_res["acc"]  
        if val_metric > self.best_val:
            self.best_val = val_metric
            self.best_payload = serialized.clone()
            improved = True

            save_dir = self.cfg.get("output", {}).get("save_dir", self.out_dir)
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"client{cid}_bestval_round{int(round_idx)}.pt")
            torch.save({"payload": self.best_payload, "val": val_res, "test": test_res}, ckpt_path)
            print(f"[Client {self.spec.cid}] New best val={val_metric:.4f} saved to {ckpt_path}")

        metrics = {
            "client": cid,
            "round": int(round_idx),
            "val_acc": float(val_res["acc"]),
            "val_loss": float(val_res["loss"]),
            "val_n": float(val_res["n"]),
            "test_acc": float(test_res["acc"]),
            "test_loss": float(test_res["loss"]),
            "test_n": float(test_res["n"]),
            "improved": bool(improved),
            "loss_trace_path": trace_path,
            "loss_evals": int(loss_evals),
            "loss_sum": float(loss_sum),
            "loss_mean": float(round_mean),
        }
        return serialized, metrics, improved

    @torch.no_grad()
    def evaluate(self, split="val") -> Dict[str, float]:
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader
        correct, total, loss_sum = 0, 0, 0.0
        for batch in loader:
            batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = self.model(**batch)
            loss = out.loss if hasattr(out, "loss") else None
            logits = out.logits if hasattr(out, "logits") else out
            if loss is not None and "labels" in batch:
                loss_sum += float(loss.item()) * int(batch["labels"].size(0))
            if "labels" in batch:
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == batch["labels"]).sum().item())
                total += int(batch["labels"].size(0))
        acc = correct / max(1, total)
        avg_loss = loss_sum / max(1, total) if total > 0 else 0.0
        return {"acc": acc, "loss": avg_loss, "n": float(total)}

    def sample_proxy_batches(self, max_batches: int = 1) -> List[Dict]:
        loader = self.sample_loader if self.sample_loader is not None else self.train_loader

        batches = []
        it = iter(loader)
        for _ in range(max_batches):
            try:
                b = next(it)
            except StopIteration:
                break
            for k, v in b.items():
                if torch.is_tensor(v) and v.ndim >= 1 and v.size(0) > 8:
                    b[k] = v[:8]
            batches.append(b)
        return batches


    def offload_and_cleanup(self):
        try:
            self.model.to("cpu")
            del self.optimizer
            if hasattr(self, "scaler"):
                del self.scaler
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
