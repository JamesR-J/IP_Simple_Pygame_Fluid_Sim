import torch

import torch.nn as nn
import torch.nn.functional as F


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()

    def act(self, input):
        policy, value = self.forward(input)  # flow the input through the nn
        return policy, value


class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)

        self.linear_a1_bn = nn.LayerNorm(num_inputs)
        self.linear_a2_bn = nn.LayerNorm(args.num_units_1)

        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

        self.nn_multiplier = 0

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)

    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = F.relu(self.linear_a1(self.linear_a1_bn(input)))
        x = F.relu(self.linear_a2(self.linear_a2_bn(x)))

        policy = self.linear_a(x) # * max_action_size # needed if actions are clamped at the output

        policy_adj = policy  # * self.nn_multiplier

        if model_original_out:
            return policy_adj, policy_adj

        return policy_adj


class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1 * 2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)

        self.linear_o_c1_bn = nn.LayerNorm(obs_shape_n)
        self.linear_a_c1_bn = nn.LayerNorm(action_shape_n)
        self.linear_c2_bn = nn.LayerNorm(args.num_units_1 * 2)

        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input, model_original_out=False):
        x_o = F.relu(self.linear_o_c1(self.linear_o_c1_bn(obs_input)))
        x_a = F.relu(self.linear_a_c1(self.linear_a_c1_bn(action_input)))

        x_cat = torch.cat([x_o, x_a], dim=1)

        x = F.relu(self.linear_c2(self.linear_c2_bn(x_cat)))

        value = self.linear_c(x)

        if model_original_out:
            return value, value

        return value
