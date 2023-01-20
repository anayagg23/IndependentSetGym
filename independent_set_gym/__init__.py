from gym.envs.registration import register

register(
    id='IndependentSet-v0',
    entry_point='independent_set_gym.envs:ind_set',
    max_episode_steps = 100
)
