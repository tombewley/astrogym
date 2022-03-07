import torch
import numpy as np
from env import AstroGymEnv
import matplotlib.pyplot as plt

RESOLUTION = 20

env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    do_render=True
    )
obs = env.reset(); env.render()
state = env._state
obs = torch.tensor(obs).unsqueeze(0).permute(0,3,1,2)

agent = torch.load("agents/deft-capybara-4/500.agent")

with torch.no_grad():
    action = torch.zeros(1, 3)
    action[0, 2] = -1

    r_pred = np.zeros((RESOLUTION, RESOLUTION))
    r_true = np.zeros((RESOLUTION, RESOLUTION))
    for i, xi in enumerate(np.linspace(-1, 1, num=RESOLUTION)):
        action[0, 0] = xi
        for j, yj in enumerate(np.linspace(-1, 1, num=RESOLUTION)):
            action[0, 1] = yj
            r_pred[i, j] = agent.net(obs, [action])[0]
            _, r_true[i, j], _, _ = env.step(action[0].cpu().numpy())
            env._state = state # Reset

            print(action, r_pred[i, j], r_true[i, j])

plt.figure()
plt.imshow(r_pred); plt.colorbar()
plt.figure()
plt.imshow(r_true); plt.colorbar()

plt.ioff(); plt.show()