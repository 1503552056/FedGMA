# server/mmoe_meta_handler.py
import torch, math
from typing import List, Dict
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.utils import SerializationTool
from factories.model_factory import build_model_and_processor

class GroupMMOEHandler(SyncParameterServerHandler):
    def __init__(self, model, global_round, sample_ratio, K=2,
                 tau=1.0, eta=0.5, epsilon=1e-6, inner_steps=1, cuda=False, logger=None):
        super().__init__(model=model, global_round=global_round, sample_ratio=sample_ratio, cuda=cuda, logger=logger)

        self.K = int(K)
        self.tau = float(tau)         
        self.eta = float(eta)          
        self.eps = float(epsilon)     
        self.inner_steps = int(inner_steps)

        self._group_logits = None      
        self._alpha = None             
        self._theta = None           
        self.group_of = None          

        self._eval_model, _ = build_model_and_processor({"task": "text-cls"}) if False else (None, None)

    # ---------- 分组 ----------
    def assign_groups(self, m: int):
        self.group_of = [(i % self.K) for i in range(m)]
        if self._group_logits is None:
            self._group_logits = torch.zeros(self.K, m)  # logits 初始化为0 -> α为均匀
            self._alpha = torch.full((self.K, m), 1.0/m)

    # ---------- 广播 ----------
    @property
    def downlink_package(self):

        return [self.model_parameters]

    def broadcast_for(self, cid: int):
        if self._theta is None:
            return [self.model_parameters]
        gid = self.group_of[cid]
        vec = self._theta[gid]
        return [vec]

    # ---------- 构建 expert pool V ----------
    @staticmethod
    def _stack_experts(serialized_list: List[torch.Tensor]) -> torch.Tensor:
        # 每个 serialized 是 1-D tensor（SerializationTool.serialize_model 输出）
        vecs = [s.view(-1).detach().cpu() for s in serialized_list]
        V = torch.stack(vecs, dim=1)  # [D, m_selected]，注意：这里只用本轮到达的，稍后我们扩展到全体 m
        return V

    # ----------θ_k = V α_k ----------
    def _compute_group_thetas(self, V_all: torch.Tensor, m_total: int):
        # V_all [D, m_total]，α [K, m_total]
        thetas = []
        for k in range(self.K):
            alpha_k = torch.softmax(self._group_logits[k] / self.tau, dim=-1)  # [m_total]
            theta_k = V_all @ alpha_k        
            thetas.append(theta_k)
        self._alpha = torch.stack([torch.softmax(self._group_logits[k]/self.tau, dim=-1) for k in range(self.K)], dim=0)
        self._theta = thetas

    # ---------- Reptile-in-α----------
    def meta_update(self, V_all: torch.Tensor, proxy_deltas: List[torch.Tensor]):
        V = V_all  # [D, m]
        VtV = (V.T @ V) + self.eps * torch.eye(V.shape[1])
        VtV_inv = torch.linalg.inv(VtV)
        Vt = V.T
        for k in range(self.K):
            if proxy_deltas[k] is None:
                continue
            delta_theta = proxy_deltas[k].to(V.dtype)  
            delta_alpha = VtV_inv @ (Vt @ delta_theta)  
            self._group_logits[k] += self.eta * delta_alpha

        self._compute_group_thetas(V_all, V_all.shape[1])
