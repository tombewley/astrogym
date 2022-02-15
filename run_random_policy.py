import torch
import matplotlib.pyplot as plt
from env import AstroGymEnv
from networks import ResNet18, MultiHeadedNetwork


BATCH_SIZE = 30


env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    do_render=False
    )

net = MultiHeadedNetwork(
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    common = ResNet18(in_channels=env.num_channels),
    head_codes = [[(131, 128), "R", (128, 1)]],
    lr=1e-4,
)
def pred_net(obs, action): 
    return net(torch.tensor(obs).permute(0,3,1,2), [torch.tensor(action)])[0]

plt.ion()
_, ax = plt.subplots()
# ax.set_xlabel("Window width"); ax.set_ylabel("Reward")
ax.set_xlabel("Batch number"); ax.set_ylabel("Sqrt(reward prediction loss)")

obs_batch, action_batch, reward_batch, batch_num = [], [], [], 0
for ep in range(1000):
    print(f"Episode {ep}")
    obs = env.reset()
    if env.do_render: env.render()
    action = env.action_space.sample()
    for t in range(10):
        
        # Random walking action
        action = 0.9 * action + 0.1 * env.action_space.sample()
        # action[2] = (action[2] - 1) / 2 # Always zoom in (a bit less boring!)
        
        
        obs_batch.append(obs)
        action_batch.append(action)
    
        obs, reward, done, info = env.step(action)
        # print(f"State: {env._state}".ljust(32), f"Reward: {reward}")

        reward_batch.append(reward * 100)
        
        if len(obs_batch) == BATCH_SIZE:
            loss = ((pred_net(obs_batch, action_batch) - torch.tensor(reward_batch).reshape(-1,1)) ** 2).sum()
            print(loss)
            net.optimise(loss)
            ax.scatter(batch_num, loss.item()**0.5, s=2, c="k")
            plt.pause(1e-3)
            obs_batch, action_batch, reward_batch = [], [], []
            batch_num += 1

        if env.do_render: env.render()

        # Plot reward against window width
        # window_width = env._state[1] - env._state[0]
        # ax.scatter(window_width, reward, s=2, c="k", alpha=0.5)
        # plt.pause(1e-6)

        if done: break

plt.ioff(); plt.show()