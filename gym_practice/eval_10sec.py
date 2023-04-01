import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

ENV_ID = 'CartPole-v1'
NUM_ENVS = [1, 2, 4, 8, 16]
NUM_EXPERIMENTS = 3
NUM_STEPS = 5000
NUM_EPISODES = 20

def evaluate(model, env, num_episodes=100):
  all_episode_rewards = []
  for i in range(num_episodes):
    episode_rewards = []
    done = False
    state = env.reset()
    while not done:
      action, _ = model.predict(state)
      state, reward, done, info = env.step(action)
      episode_rewards.append(reward)
    all_episode_rewards.append(sum(episode_rewards))

  return np.mean(all_episode_rewards)

def main():
  reward_averages = []
  reward_std = []
  training_times = []
  total_env = 0
  for num_envs in NUM_ENVS:
    total_env += num_envs
    print('process:', num_envs)

    # dummy
    if num_envs == 1:
      train_env = make_vec_env(
        ENV_ID,
        n_envs=num_envs,
        seed=[i+total_env for i in range(num_envs)],
        vec_env_cls=DummyVecEnv
      )
    # subproc
    else:
      train_env = make_vec_env(
        ENV_ID,
        n_envs=num_envs,
        seed=[i+total_env for i in range(num_envs)],
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={'start_method': 'spawn'}
      )

    eval_env = make_vec_enc(
      ENV_ID,
      n_envs=1,
      seed=0,
      vec_env_cls=DummyVecEnv
    )

    # calc steps of 10 sec
    train_env.reset()
    model = PPO('MlpPolicy', train_env, verbose=0)
    start = time.time()
    model.learn(total_timesteps=NUM_STEPS)
    steps_per_sec = NUM_STEPS / (time.time() - start)
    num_steps.append(int(steps_per_sec*10))

    rewards = []
    times = []
    for experiment in range(NUM_EXPERIMENTS):
      train_env.reset()
      model = PPO('MlpPolicy', train_env, verbose=0)
      start = time.time()
      model.learn(total_timesteps=int(steps_per_sec*10))
      times.append(time.time() - start)

      mean_reward = evaluate(model, eval_env, num_episodes=NUM_EPISODES)
      reward.append(mean_reward)

    train_env.close()
    eval_env.close()

    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

  plt.errorbar(NUM_EVNS, reward_averages, yerr=reward_std, capsize=2)
  plt.xlabel('number of envs')
  plt.ylabel('mean reward')
  plt.show()

  plt.bar(range(len(NUM_ENV)), num_steps)
  plt.xticks(range(len(NUM_ENVS)), NUM_ENVS)
  plt.xlabel('number of envs')
  plt.ylabel('number of steps')
  plt.show()

if __name__ == "__main__":
  main()
