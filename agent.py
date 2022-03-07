from networks import ResNet18, MultiHeadedNetwork

import torch
import torch.nn.functional as F
import numpy as np
from rlutils.agents._generic import Agent
from rlutils.common.memory import ReplayMemory


CURRENT = False


class ResNetAgent(Agent):
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        self.net = MultiHeadedNetwork(
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            common = ResNet18(in_channels=self.env.num_channels),
            head_codes = [[(128 if CURRENT else 131, 128), "R", (128, 1)]],
            lr=self.P["lr"],
        )
        self.start()

    def start(self):
        self.memory = ReplayMemory(self.P["replay_capacity"])
        self.total_t = 0
        self.ep_losses = []

    def act(self, obs, explore, do_extra):
        action = self.env.action_space.sample()
        # action[2] = (action[2] - 1) / 2 # <<< NOTE: Always zoom in!
        return action, {}

    def update_on_batch(self):
        obs, action, reward, _, next_obs = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
        if obs is None: return 
        reward_pred = self.net(obs.permute(0,3,1,2), [None if CURRENT else action])[0]   
        loss = F.smooth_l1_loss(reward_pred, reward.reshape(-1,1))
        # loss = ((reward_pred - reward.reshape(-1,1)) ** 2).sum()
        self.net.optimise(loss)
        return loss.item(),

    def per_timestep(self, obs, action, reward, next_obs, done):
        self.memory.add(obs, action, reward, next_obs, done)  
        self.total_t += 1
        if self.total_t % self.P["update_freq"] == 0:
            losses = self.update_on_batch()
            if losses: self.ep_losses.append(losses)
        
    def per_episode(self):
        if self.ep_losses: 
            mean_value_loss, = np.nanmean(self.ep_losses, axis=0)
            del self.ep_losses[:]
            return {"value_loss": mean_value_loss}
        return {}