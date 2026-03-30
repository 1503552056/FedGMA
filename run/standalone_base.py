# run/standalone_base.py
from __future__ import annotations
import abc, random, time
from typing import Dict, List, Any

class StandalonePipeline(abc.ABC):


    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.rounds = cfg["runtime"]["rounds"]
        self.num_clients = cfg["runtime"]["num_clients"]
        self.clients_per_round = cfg["runtime"]["clients_per_round"]
        self.seed = cfg["runtime"]["seed"]

        self.server = self.build_server()
        self.clients = self.build_clients()
        assert len(self.clients) == self.num_clients, \
            f"clients built = {len(self.clients)}, expected = {self.num_clients}"

    @abc.abstractmethod
    def build_server(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build_clients(self) -> List[Any]:
        raise NotImplementedError

    def sample_clients(self, round_idx: int) -> List[int]:
        ids = list(range(self.num_clients))
        random.Random(self.seed + round_idx).shuffle(ids)
        return ids[: self.clients_per_round]

    def make_payloads(self) -> Dict[int, Dict[str, Any]]:
        return self.server.broadcast() if hasattr(self.server, "broadcast") else {}

    def dispatch(self, selected_ids: List[int], payloads: Dict[int, Dict[str, Any]]) -> List[Dict]:
        messages = []
        for cid in selected_ids:
            ok = self.clients[cid].local_process(payloads.get(cid, {}))
            if ok:
                messages.append(self.clients[cid].upload())
        return messages

    def on_round_start(self, round_idx: int): pass
    def on_after_broadcast(self, round_idx: int, payloads: Dict[int, Dict]): pass
    def on_after_local(self, round_idx: int, messages: List[Dict]): pass
    def on_after_aggregate(self, round_idx: int): pass
    def should_early_stop(self, round_idx: int) -> bool: return False

    # ---------- 主循环 ----------
    def run(self):
        for r in range(self.rounds):
            t0 = time.time()
            self.on_round_start(r)

            selected = self.sample_clients(r)
            payloads = self.make_payloads()
            self.on_after_broadcast(r, payloads)

            messages = self.dispatch(selected, payloads)
            self.on_after_local(r, messages)

            if hasattr(self.server, "global_update"):
                self.server.global_update(messages)
            self.on_after_aggregate(r)

            dt = time.time() - t0
            print(f"[Round {r}] selected={len(selected)} messages={len(messages)} time={dt:.2f}s")

            if self.should_early_stop(r):
                print(f"[Early stop] at round {r}")
                break
