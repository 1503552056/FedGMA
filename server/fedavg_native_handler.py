import torch
from fedlab.core.server.handler import SyncParameterServerHandler

class FedAvgNativeServerHandler(SyncParameterServerHandler):
    def __init__(self, model, global_round, sample_ratio, cuda=False, logger=None):
        super().__init__(model=model,
                         global_round=global_round,
                         sample_ratio=sample_ratio,
                         cuda=cuda,
                         logger=logger)

    @property
    def downlink_package(self):
        return [self.model_parameters]  

    @property
    def if_stop(self):
        return super().if_stop  
