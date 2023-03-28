import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('CatPole-v1')
env = DummyVecEnv([lambda: env]) #(1)

# Generate model (2)
model = PPO2('MlpPolicy', env, verbose=1)

# Learn model (3)
model.learn(total_timesteps=100000)

# Test model
state = env.reset()
for i in range(200):
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
