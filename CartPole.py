# # Import dependencies
import os
import gym  # OpenAI gym library for RL environments
from stable_baselines3 import PPO  # Import the PPO algorithm
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)  # wrap the environment to make it parallel and vectorized
from stable_baselines3.common.evaluation import evaluate_policy  # Evaluate the agent

# Environment name
name = "CartPole-v1"
env = gym.make(name)


# Train an RL model
def environment_setup(env):
    # Create the environment using the gym library
    # we use DummyVecEnv to wrap the environment to make it parallel and vectorized.
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1)
    return model


def train_model(model, total_timesteps):
    model.learn(total_timesteps=total_timesteps)
    return model


def save_model(model, path):
    model.save(path)


def load_model(path, env):
    model = PPO.load(path, env=env)
    return model


def evaluate(model, env, n_eval_episodes=10):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            print("info", info)
            break


# Testing
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()
#     if done:
#         print("info", info)
#         break


def test_model(model, env):
    episodes = 5
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))
    env.close()


model = environment_setup(env)
model = train_model(model, 20000)

# Save and Reload Model
PPO_path = os.path.join("Training", "Saved_models", "PPO_model")
save_model(model, PPO_path)
# del model
model = load_model(PPO_path, env=env)

#  Evaluation
from stable_baselines3.common.evaluation import evaluate_policy

evaluate_policy(model, env, n_eval_episodes=10, render=True)
# env.close()

# Testing
test_model(model, env)

# Viewing Logs in Tensorboard ====================================
log_path = os.path.join("Training", "Logs")

training_log_path = os.path.join(log_path, "PPO_3")
# RUN ON terminal:>> !tensorboard --logdir={training_log_path}


# Adding a callback to the training Stage=======================
# from stable_baselines3.common.callbacks import (
#     EvalCallback,
#     StopTrainingOnRewardThreshold,
# )

# save_path = os.path.join("Training", "Saved Models")
# log_path = os.path.join("Training", "Logs")

# env = gym.make(name)
# env = DummyVecEnv([lambda: env])

# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
# eval_callback = EvalCallback(
#     env,
#     callback_on_new_best=stop_callback,
#     eval_freq=10000,
#     best_model_save_path=save_path,
#     verbose=1,
# )
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=20000, callback=eval_callback)
# model_path = os.path.join("Training", "Saved Models", "best_model")
# model = PPO.load(model_path, env=env)
# evaluate_policy(model, env, n_eval_episodes=10, render=True)
# env.close()


# Changing Policies ==============================================
# net_arch = [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs={"net_arch": net_arch})
# model.learn(total_timesteps=20000, callback=eval_callback)


# Using an Alternate Algorithm ===================================

# from stable_baselines3 import DQN

# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=20000, callback=eval_callback)
# dqn_path = os.path.join("Training", "Saved Models", "DQN_model")
# model.save(dqn_path)
# model = DQN.load(dqn_path, env=env)
# evaluate_policy(model, env, n_eval_episodes=10, render=True)
# env.close()
