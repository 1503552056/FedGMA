# run/standalone_fedavg.py
import argparse, os, yaml, torch
from typing import List

from run.standalone_base import StandalonePipeline
from server.FedAvgServerHandler import FedAvgServerHandler
from client.GenericClientTrainer import GenericClientTrainer

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class FedAvgStandalone(StandalonePipeline):
    def build_server(self):
        return FedAvgServerHandler(torch.nn.Linear(2, 2), self.cfg)

    def build_clients(self) -> List[GenericClientTrainer]:
        return [GenericClientTrainer(self.cfg, cid) for cid in range(self.cfg["runtime"]["num_clients"])]

    def on_round_start(self, round_idx: int):
        if round_idx == 0:
            print("[Init] FedAvgStandalone started")

    def on_after_aggregate(self, round_idx: int):
        gv = getattr(self.server, "global_vec", None)
        if gv is not None:
            print(f"[Round {round_idx}] global_vec_dim={gv.numel()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_task.yaml")
    args = ap.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)
    cfg = load_yaml(args.config)

    app = FedAvgStandalone(cfg)
    app.run()

if __name__ == "__main__":
    main()
