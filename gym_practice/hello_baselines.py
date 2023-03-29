import gymnasium as gym
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines import PPO2
from stable_baselines3 import PPO

# env = gym.make('CartPole-v1')
env_id = 'CartPole-v1'
# env = DummyVecEnv([lambda: env]) #(1)
env = make_vec_env(env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv)

# Generate model (2)
model = PPO('MlpPolicy', env, verbose=1)

# Learn model (3)
model.learn(total_timesteps=100000)

# Test model
state = env.reset()
for i in range(2000):
  # render environment
  env.render()

  # predict model (4)
  action, _ = model.predict(state, deterministic=True)

  # action 1 step
  state, rewards, done, info = env.step(action)

  # complete episode
  if done:
    break

# close environment
env.close()
