import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from maddpg.actor_critic import Critic
import numpy as np
import common.utils as me


class MetaLearner:

    def __init__(self, args, sampler, gamma=0.95, outer_lr=1e-8, tau=1.0, device='cpu'):
        self.task_sampler = sampler
        self.gamma = gamma
        self.outer_lr = outer_lr
        self.tau = tau
        # self.to(device)

        self.centralized_q = Critic(args=args)
        self.target_centralized_q = Critic(args=args)
        self.centralized_q_optim = torch.optim.Adam(self.centralized_q.parameters(), lr=self.outer_lr)

        args.scenario_name = "simple_reference"
        _, args = me.make_env(args=args)

        self.args = args
        self.train_step = 0
        self.total_training_step = 20000
        self.outer_times = 1000
        self.episode_limit = 100
        self.num_tasks = 4
        self.save_rate = 10

    def train(self):
        result = []
        for i in range(self.total_training_step):
            print("Meta Training " + str(i + 1) + " sampling " + str(self.num_tasks) + " tasks")
            tasks = self.task_sampler.sample(num_tasks=self.num_tasks)
            inner_a_loss = []
            inner_q_loss = []
            inner_return = []
            for j, t in enumerate(tasks):
                # inner training
                print("\tIn task " + str(j + 1) + " sampling " + str(self.task_sampler.batch_size) + " trajectory")
                # load generalized centralized q function
                for a in t.agents:
                    a.policy.critic_target_network.load_state_dict(self.centralized_q.state_dict())
                # inner training
                r, a_loss, q_loss = t.run()
                print("\tCompleted task " + str(j + 1), "validation_return: " + str(r), "a_loss: " + str(a_loss),
                      "q_loss: " + str(q_loss))
                inner_a_loss.append(a_loss)
                inner_q_loss.append(q_loss)
                inner_return.append(r)

            # outer training
            for time_step in range(self.outer_times):
                centralized_q_loss = None
                # reset the environment
                for j, t in enumerate(tasks):
                    if time_step % t.episode_limit == 0:
                        s = t.env.reset()
                    u = []
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(t.agents):
                            action = agent.select_action(s[agent_id], t.noise, t.epsilon)
                            u.append(action)
                            actions.append(action)
                    for _ in range(t.args.n_agents, t.args.n_players):
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    s_next, r, done, info = t.env.step(actions)
                    t.buffer.store_episode(s[:t.args.n_agents], u, r[:t.args.n_agents],
                                           s_next[:t.args.n_agents])
                    s = s_next
                    if t.buffer.current_size >= t.args.batch_size:
                        transitions = t.buffer.sample(t.args.batch_size)
                        for idx, agent in enumerate(t.agents):
                            other_agents = t.agents.copy()
                            other_agents.remove(agent)
                            # agent.learn(transitions, other_agents)

                            for key in transitions.keys():
                                transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
                            r = transitions['r_%d' % agent.policy.agent_id]  # 训练时只需要自己的reward
                            o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
                            for agent_id in range(agent.policy.args.n_agents):
                                o.append(transitions['o_%d' % agent_id])
                                u.append(transitions['u_%d' % agent_id])
                                o_next.append(transitions['o_next_%d' % agent_id])

                            # calculate the target Q value function
                            u_next = []
                            with torch.no_grad():
                                # 得到下一个状态对应的动作
                                index = 0
                                for agent_id in range(agent.policy.args.n_agents):
                                    if agent_id == agent.policy.agent_id:
                                        u_next.append(agent.policy.actor_target_network(o_next[agent_id]))
                                    else:
                                        # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                                        u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                                        index += 1
                                q_next = agent.policy.critic_network(o_next, u_next).detach()

                                target_q = (r.unsqueeze(1) + agent.policy.args.gamma * q_next).detach()

                            # the q loss
                            q_value = self.centralized_q(o, u)
                            q_loss = (target_q - q_value).pow(2).mean()
                            # the actor loss
                            # 重新选择联合动作中当前agent的动作，其他agent的动作不变
                            u[agent.policy.agent_id] = agent.policy.actor_network(o[agent.policy.agent_id])
                            actor_loss = - agent.policy.critic_network(o, u).mean()
                            if centralized_q_loss is None:
                                centralized_q_loss = q_loss
                            else:
                                centralized_q_loss = centralized_q_loss.add(q_loss)
                    t.noise = max(0.05, t.noise - 0.0000005)
                    t.epsilon = max(0.05, t.epsilon - 0.0000005)
                self.centralized_q_optim.zero_grad()
                centralized_q_loss.backward()
                self.centralized_q_optim.step()
                for target_param, param in zip(self.target_centralized_q.parameters(),
                                               self.centralized_q.parameters()):
                    target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
            to_save = [i + 1, np.mean(inner_return), np.mean(inner_a_loss), np.mean(inner_q_loss),
                       centralized_q_loss.item()]
            result.append(to_save)

            if i % self.save_rate == 0:
                print("Saving training information and meta centralized q function parameters", end=" ")
                np.save("./MAML_result/training_info.npy", np.array(result))
                torch.save(self.centralized_q.state_dict(), './MAML_result/centralized_q_params.pth')
                print("and successfully saved")

            print("Meta Update: " + str(i + 1), "\n\tinner_batch_avg_validation_return: " + str(np.mean(inner_return)),
                  "\n\tinner_batch_average_a_loss: " + str(np.mean(inner_a_loss)),
                  "\n\tinner_batch_average_q_loss: " + str(np.mean(inner_q_loss)),
                  "\n\tmeta_centralized_q_loss: " + str(centralized_q_loss.item()))
