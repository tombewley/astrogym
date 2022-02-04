import gym
from env import AstroGymEnv

env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    do_render=True
    )

for ep in range(10):
    obs = env.reset()
    action = env.action_space.sample()
    for t in range(20):
        # Random walking action
        action = 0.9 * action + 0.1 * env.action_space.sample()
        action[2] = (action[2] - 1) / 2 # Always zoom in (a bit less boring!)
        obs, reward, done, info = env.step(action)
        render = env.render()
        if done: break