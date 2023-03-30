import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

env_id = 'CartPole-v1'
env = make_vec_env(
  env_id,
  n_envs=1,
  seed=0,
  vec_env_cls=DummyVecEnv
)

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=100000)

model.save('model_sample')
