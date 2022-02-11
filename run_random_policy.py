import torch
import matplotlib.pyplot as plt
from env import AstroGymEnv
from networks import ResNet18


env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    do_render=True
    )

net = ResNet18(in_channels=env.num_channels)
def pred_resnet(obs): 
    with torch.no_grad():
        return net(torch.tensor(obs).unsqueeze(0).permute(0,3,1,2))

_, ax = plt.subplots()
ax.set_xlabel("Window width"); ax.set_ylabel("Reward")

for ep in range(10):
    print(f"Episode {ep}")
    obs = env.reset()
    # print("ResNet output from initial observation:", pred_resnet(obs))
    if env.do_render: env.render()
    action = env.action_space.sample()
    for t in range(10):
        
        # Random walking action
        action = 0.9 * action + 0.1 * env.action_space.sample()
        action[2] = (action[2] - 1) / 2 # Always zoom in (a bit less boring!)
        obs, reward, done, info = env.step(action)
        print(f"State: {env._state}".ljust(32), f"Reward: {reward}")
        if env.do_render: env.render()

        # Plot reward against window width
        window_width = env._state[1] - env._state[0]
        ax.scatter(window_width, reward, s=2, c="k", alpha=0.5)

        if done: break

plt.show()