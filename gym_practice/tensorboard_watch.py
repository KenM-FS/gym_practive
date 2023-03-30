import gymnasium as gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

env = make_vec_env(
  'CartPole-v1',
  n_envs=1,
  seed=0,
  vec_env_cls=DummyVecEnv
)

model = PPO(
  'MlpPolicy',
  env,
  verbose=1,
  tensorboard_log=log_dir
)

model.learn(total_timesteps=100000)

state = env.reset()
for i in range(5000):
  env.render()
  action, _ = model.predict(state, deterministic=True)
  state, reward, done, info = env.step(action)
  if done:
    break

env.close()
