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

model = PPO.load('model_sample')

state = env.reset()
for i in range(5000):
  env.render()
  action, _ = model.predict(state, deterministic=True)
  state, reward, done, info = env.step(action)
  if done:
    break

env.close()
