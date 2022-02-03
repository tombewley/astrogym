import gym
from env import AstroGymEnv

env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    do_render=True
    )

for ep in range(10):
    obs = env.reset()
    action = env.action_space.sample()
    for t in range(10):
        # Random walking action
        action = 0.9 * action + 0.1 * env.action_space.sample()
        obs, reward, done, info = env.step(action)
        render = env.render()
        if done: break