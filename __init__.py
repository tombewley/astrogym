from gym.envs.registration import register

register(
    id="AstroGym-v0", 
    entry_point="astrogym.env:AstroGymEnv",
	)