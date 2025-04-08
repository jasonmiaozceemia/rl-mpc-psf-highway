from gymnasium.envs.registration import register
from my_envs.custom_highway import MyHighwayEnv


register(
    id="my-highway-v0",
    entry_point="my_envs.custom_highway:MyHighwayEnv",
)