import numpy as np
import torch
import torch.nn.functional as F


class BatchEpisodes:
    def __init__(self, num_agents, batch_size, gamma=0.95, device="cpu"):
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observation_list = [[] for _ in range(self.batch_size)]
        self._action_list = [[] for _ in range(self.batch_size)]
        self._reward_list = [[] for _ in range(self.batch_size)]

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None

    def observations(self):
        pass

    def actions(self):
        pass

    def rewards(self):
        pass

    def returns(self):
        pass

    def append(self, observations, actions, rewards, batch_ids):
        for o, a, r, index in zip(observations, actions, rewards, batch_ids):
            if index is None:
                continue
            self._observation_list[index].append(o.astype(np.float32))
            self._action_list[index].append(a.astype(np.float32))
            self._reward_list[index].append(r.astype(np.float32))
