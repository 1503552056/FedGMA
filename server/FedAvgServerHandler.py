import torch
from fedlab.core.server.handler import ServerHandler
class FedAvgServerHandler(ServerHandler):
    def __init__(self, model, cfg):
        super().__init__(model)
        self.cfg = cfg
        self.global_vec = None
        self.meta = None
    def parse_messages(self, messages):
        vecs = []
        for msg in messages:
            v = torch.tensor(msg['vec']).float()
            if self.meta is None:
                self.meta = msg.get('meta', None)
            vecs.append(v)
        if len(vecs) == 0:
            return None
        return torch.stack(vecs, dim=0)
    def global_update(self, messages):
        stacked = self.parse_messages(messages)
        if stacked is None:
            return
        avg = stacked.mean(dim=0)
        if self.global_vec is None:
            self.global_vec = avg
        else:
            self.global_vec = 0.5 * self.global_vec + 0.5 * avg
    def broadcast(self):
        payloads = {}
        for cid in range(self.cfg['runtime']['num_clients']):
            payloads[cid] = {'global_vec': None if self.global_vec is None else self.global_vec.cpu().numpy()}
        return payloads
