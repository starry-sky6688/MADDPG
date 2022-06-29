import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector


class MetaLearner:

    def __init__(self, args, agent_id, sampler, policy, baseline, gamma=0.95, fast_lr=0.5, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

        self.args = args
        self.agent_id = agent_id
        self.train_step = 0