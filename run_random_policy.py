import gym
from env import AstroGymEnv
import torch
from resnet import ResNet18

env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    do_render=True
    )

net = ResNet18()

def pred_resnet(obs): 
    # NOTE: It seems that the resnet can only accept a single channel?
    with torch.no_grad():
        return net(torch.tensor(obs[:,:,0:1]).unsqueeze(0).permute(0,3,1,2))

for ep in range(10):
    obs = env.reset()
    print("ResNet output from initial observation:", pred_resnet(obs))
    env.render()
    action = env.action_space.sample()
    for t in range(10):
        # Random walking action
        action = 0.9 * action + 0.1 * env.action_space.sample()
        action[2] = (action[2] - 1) / 2 # Always zoom in (a bit less boring!)
        obs, reward, done, info = env.step(action)
        print("State:\t", env._state, "\tReward:\t", reward)
        env.render()
        if done: break