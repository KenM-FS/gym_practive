import gymnasium as gym
import os
import numpy as np
import datetime
import pytz # timezone
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy

num_update = 0
beat_mean_reward = -np.inf

def callback(_locales, _globals):
  global num_update
  global best_mean_reward

  if (num_update + 1) % 100 == 0:
    _, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(y) > 0:
      mean_reward = np.mean(y[-100:])
      update_model = mean_reward > best_mean_reward
      if update_model:
        best_mean_reward = mean_reward
        _locals['self'].save('best_model')

      print("time: {}, num_update: {}, mean: {:.2f}, beat_mean: {:.2f}, model_update {}".format(
        datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
        num_update,
        mean_reward,
        beat_mean_reward,
        update_model
      ))
  num_update += 1
  return True

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

env = make_vec_env(
  'CartPole-v1',
  n_envs=1,
  seed=0,
  vec_env_cls=DummyVecEnv,
  monitor_dir=log_dir,
  monitor_kwargs={'allow_early_resets': True}
)

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=100000, callback=callback)
