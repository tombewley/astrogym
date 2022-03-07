import torch
import matplotlib.pyplot as plt
from heuristics import *

BASELINE = 0.17008188063947

agent = torch.load("agents/efficient-breeze-6/500.agent")

obs = []

# Near-black
obs.append(0.01 * torch.ones(1, 5, 100, 100))

# White
obs.append(torch.ones(1, 5, 100, 100))

# Bright square
obs.append(torch.zeros(1, 5, 100, 100))
obs[-1][:,:,20:80,20:80] = 1

# Several small squares
obs.append(torch.zeros(1, 5, 100, 100))
obs[-1][:,:,20:40,20:40] = 1
obs[-1][:,:,20:40,60:80] = 1
obs[-1][:,:,60:80,20:40] = 1
obs[-1][:,:,60:80,60:80] = 1

# Random
obs.append(torch.rand(1, 5, 100, 100))

# Thresholded random
obs.append(torch.round(torch.rand(1, 5, 100, 100)))

# Less random
obs.append(torch.rand(1, 5, 100, 100))
obs[-1] = 0.4 + (obs[-1] * 0.2)

with torch.no_grad():
    for o in obs:
        r_true = hoyer_torch(o, BASELINE)
        r_pred = agent.net(o, [None])[0].item()
        plt.figure()
        plt.imshow(o[0,1:4].permute(1,2,0), vmin=0, vmax=1)
        plt.title(f"{r_true} | {r_pred}")

plt.show()