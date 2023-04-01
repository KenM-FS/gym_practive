import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

ENV_ID = 'CartPole-v1'
NUM_ENV = 4

# Can be replaced by make_vec_env
# def make_env(env_id, rank, seed=0):
#   def _init():
#     env = gym.make(env_id)
#     env.seed(seed + rank)
#     return env
#   set_global_seeds(seed)
#   return _init

def main():
  # train_env = SubprocVecEnv([make_env(ENV_ID, i) for i in range(NUM_ENV)])
  tain_env = make_ven_env(
    ENV_ID,
    n_envs=NUM_ENV,
    seed=0,
    vec_env_cls=SubprocVecEnv
  )

  model = PPO('MlpPolicy', train_env, verbose=1)

  model.learn(total_timesteps=100000)

  test_env = make_vec_env(
    ENV_ID,
    n_envs=1,
    seed=0,
    vec_env_cls=DummyVecEnv
  )

  state = test_env.reset()
  for i in range(200):
    test_env.render()

    action, _ = model.predict(state)

    state, rewards, done, info = test_env.step(action)

    if done:
      break

  env.close()

if __name__ == "__main__":
  main()
