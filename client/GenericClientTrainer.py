import torch
from fedlab.core.client.trainer import ClientTrainer
from factories.model_factory import build_model_and_processor
from factories.data_factory import build_dataloaders
from utils.delta_codec import flatten_state_dict
class GenericClientTrainer(ClientTrainer):
    def __init__(self, cfg, client_id: int):
        super().__init__(model=None)
        self.cfg = cfg
        self.client_id = client_id
        self.model, self.proc = build_model_and_processor(cfg)
        self.train_loader, self.eval_loader = build_dataloaders(cfg, self.proc, client_id)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=cfg['train']['lr'], weight_decay=cfg['train']['wd'])
        self._named_params = [n for n,_ in self.model.named_parameters() if _.requires_grad]
        self._base_state = {n: p.detach().clone() for n,p in self.model.named_parameters() if n in self._named_params}
    def _batch_to_device(self, batch):
        if isinstance(batch, dict):
            return {k: (v.to(self.device) if hasattr(v, 'to') else v) for k,v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [x.to(self.device) if hasattr(x, 'to') else x for x in batch]
        return batch
    def _forward_loss(self, batch):
        if isinstance(batch, dict):
            out = self.model(**batch)
            return out.loss
        else:
            x,y = batch
            logits = self.model(x)
            return torch.nn.functional.cross_entropy(logits, y)
    def local_process(self, payload):
        steps = 0
        self.model.train()
        for _ in range(self.cfg['train']['local_epochs']):
            for batch in self.train_loader:
                batch = self._batch_to_device(batch)
                loss = self._forward_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                steps += 1
                if self.cfg['train']['max_steps'] and steps >= self.cfg['train']['max_steps']:
                    break
            if self.cfg['train']['max_steps'] and steps >= self.cfg['train']['max_steps']:
                break
        cur = {n: p.detach().clone() for n,p in self.model.named_parameters() if n in self._named_params}
        for n in cur:
            cur[n] = cur[n] - self._base_state[n]
        vec, meta = flatten_state_dict(cur)
        self._upload_msg = {'vec': vec.cpu().numpy(), 'meta': meta, 'client_id': self.client_id}
        self._base_state = {n: p.detach().clone() for n,p in self.model.named_parameters() if n in self._named_params}
        return True
    def upload(self):
        return self._upload_msg
