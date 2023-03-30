import gymnasium as gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
# from stable_baselines.bench import Monitor
from stable_baselines3.common.monitor import Monitor

# Make log directory (1)
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# Make env
env_id = 'CartPole-v1'
env = make_vec_env(
  env_id,
  n_envs=1,
  seed=0,
  vec_env_cls=DummyVecEnv,
  monitor_dir=log_dir,
  monitor_kwargs={'allow_early_resets': True}
)
# env = Monitor(env, log_dir, allow_early_resets=True)

# Generate model
model = PPO('MlpPolicy', env, verbose=1)

# Learn model
model.learn(total_timesteps=10000)

# Test model
state = env.reset()
for i in range(2000):
  # render environment
  env.render()

  # predict model
  action, _ = model.predict(state, deterministic=True)

  # action 1 step
  state, rewards, done, info = env.step(action)

  # complete episode
  if done:
    break

# close env
env.close()
