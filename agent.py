from networks import ResNet18, MultiHeadedNetwork

import torch
import torch.nn.functional as F
import numpy as np
from rlutils.agents._generic import Agent


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
        self.clear()
        self.ep_losses = []

    def act(self, obs, explore, do_extra):
        action = self.env.action_space.sample()
        action[2] = (action[2] - 1) / 2 # <<< NOTE: Always zoom in!
        return action, {}

    def per_timestep(self, obs, action, reward, next_obs, done):
        if CURRENT: 
            self.obs_batch.append(next_obs) 
        else:
            self.obs_batch.append(obs)
            self.action_batch.append(action)
        self.reward_batch.append(reward)
        if len(self.obs_batch) == self.P["batch_size"]:
            if CURRENT:
                reward_pred = self.net(torch.cat(self.obs_batch).permute(0,3,1,2), [None])[0]
            else:
                reward_pred = self.net(torch.cat(self.obs_batch).permute(0,3,1,2), 
                                      [torch.tensor(np.array(self.action_batch), device=self.device)])[0]

            loss = F.smooth_l1_loss(reward_pred, torch.tensor(self.reward_batch, device=self.device).reshape(-1,1))
            # loss = ((reward_pred - torch.tensor(self.reward_batch, device=self.device).reshape(-1,1)) ** 2).sum()
            self.net.optimise(loss)

            print(reward_pred[-1].item(), self.reward_batch[-1], loss.item())
            self.clear()
            self.ep_losses.append([loss.item()])
        
    def per_episode(self):
        if self.ep_losses: 
            mean_value_loss, = np.nanmean(self.ep_losses, axis=0)
            del self.ep_losses[:]
            return {"value_loss": mean_value_loss}
        return {}

    def clear(self): 
        self.obs_batch, self.action_batch, self.reward_batch = [], [], []
