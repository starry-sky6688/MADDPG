import torch
import torch.nn as nn
import torch.nn.functional as F
import common.utils as me
from multiagent.multi_discrete import MultiDiscrete
import copy

class Centralized_q(nn.Module):
    def __init__(self, args, task_sampler):
        super(Centralized_q, self).__init__()
        self.max_action = args.high_action if hasattr(args, "high_action") else 1
        Args=[]
        for scenario in task_sampler.scenarios_names:
            arg = copy.copy(args)
            arg.scenario_name = scenario
            env, arg = me.make_env(args=arg)
            Args.append(arg)

        Shape=[]
        for a in Args:
            Shape.append(sum(a.obs_shape) + sum(a.action_shape))

        input_shape = max(Shape)
        self.input_shape=input_shape
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        while True:
            if len(x[0])==self.input_shape:
                break
            x = torch.cat((x, torch.tensor([[0]]*256)), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
