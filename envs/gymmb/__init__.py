from gym.envs.registration import register

register(
    id='GYMMB_HalfCheetah-v2',
    entry_point='envs.gymmb.cheetah:GYMMB_HalfCheetah',
    max_episode_steps=1000
)

register(
    id='GYMMB_Pusher-v2',
    entry_point='envs.gymmb.pusher:GYMMB_Pusher',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='GYMMB_Hopper-v2',
    entry_point='envs.gymmb.hopper:GYMMB_Hopper',
    max_episode_steps=1000
)

register(
    id='GYMMB_Walker2d-v2',
    entry_point='envs.gymmb.walker2d:GYMMB_Walker2d',
    max_episode_steps=1000
)

register(
    id='GYMMB_Swimmer-v2',
    entry_point='envs.gymmb.swimmer:GYMMB_Swimmer',
    max_episode_steps=1000
)

register(
    id='GYMMB_Pendulum-v0',
    entry_point='envs.gymmb.pendulum:GYMMB_Pendulum',
    max_episode_steps=200
)

register(
    id='GYMMB_CartPole-v1',
    entry_point='envs.gymmb.cartpole:GYMMB_ContinuousCartPole',
    max_episode_steps=500
)

register(
    id='GYMMB_Test-v0',
    entry_point='envs.gymmb.cheetah:GYMMB_HalfCheetah',
    max_episode_steps=100
)

