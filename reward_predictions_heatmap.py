import torch
import numpy as np
from env import AstroGymEnv
import matplotlib.pyplot as plt

RESOLUTION = 10

env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    do_render=True
    )
obs = env.reset(); env.render()
obs = torch.tensor(obs).unsqueeze(0).permute(0,3,1,2)

agent = torch.load("agents/olive-bush-3/500.agent")

with torch.no_grad():
    action = torch.zeros(1, 3)
    action[0, 2] = -1

    r_pred = np.zeros((RESOLUTION, RESOLUTION))
    for i, xi in enumerate(np.linspace(-1, 1, num=RESOLUTION)):
        action[0, 0] = xi
        for j, yj in enumerate(np.linspace(-1, 1, num=RESOLUTION)):
            action[0, 1] = yj
            r_pred[i, j] = agent.net(obs, [action])[0]
            print(action, r_pred[i, j])

plt.figure()
plt.imshow(r_pred)

plt.ioff(); plt.show()