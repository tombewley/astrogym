from env import AstroGymEnv
from agent import ResNetAgent
from heuristics import *
from rlutils import train


DP = {
    "project_name": "astrogym",
    "wandb_monitor": True,

    "reward": hoyer_numpy,

    "num_episodes": 500,
    "checkpoint_freq": 500,
    "episode_time_limit": 10
}
AP = {
    "lr": 1e-4,
    "replay_capacity": 10000,
    "batch_size": 32,
    "update_freq": 1,
}

env = AstroGymEnv( 
    img="images/mc_channelwise_clipping.npy",
    reward=DP["reward"],
    do_render=False
    )
train(ResNetAgent(env, AP), DP)