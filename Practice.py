# import gym
# import numpy as np
# import time

# SEED = 42

# env = gym.make("CartPole-v1", render_mode="human")
# observation, info = env.reset(seed=SEED)

# lastFaliure = 0
# for i in range(1000):
#     action = np.random.randint(0, env.action_space.n)
#     observation, reward, done, truncated, info = env.step(action)
#     if done:
#         # time.sleep(0.5)
#         observation, info = env.reset()
#         print(f"Episode terminated after {i-lastFaliure} steps. Resetting environment.")
#         lastFaliure = i
# env.close()


# # Import dependencies
# import os
import gym  # OpenAI gym library for RL environments
from stable_baselines3 import PPO  # Import the PPO algorithm
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)  # wrap the environment to make it parallel and vectorized
from stable_baselines3.common.evaluation import evaluate_policy  # Evaluate the agent


# # Load Environment

# # Case sensitive environment name, maping to pre-installed environments
environment_name = "CartPole-v1"

# # Create the environment using the gym library
env = gym.make(environment_name, render_mode="human")


# episodes to test the environment 5 times.
# Think of episodes as the number of games we play to test the environment. Some environment have a fixed episode length, while others are continuous.
# E.g. CartPole has 200 Frames. Others are continuous like breakout, play until you lose all lives.
episodes = 5
# frames = []
# Loop through each episode (game)
for episode in range(1, episodes + 1):
    # Reset the state of the environment. Observations for the environment.
    state = env.reset()
    # weather or not the episode (game) is done
    done = False
    # Score counter of the game
    score = 0

    # While the episode is not done, continue playing the game
    while not done:
        # Render the environment to the screen> It allows us to see the game being played or the graphical representation of the game.
        # env.render(mode="human")
        # frames.append(env.render())

        # Take a random action in the environment. The action space is the number of actions the agent can take in the environment.
        # E.g. CartPole has 2 actions, move left or move right. Discrete action space (0,1).
        action = env.action_space.sample()

        # Note:
        # You can check env.observation_space: Box(4,) for CartPole.
        # Observation space is the number of observations the agent can make in the environment. It allow the agent to make decisions based on the observations.

        # env.step takes the action and returns the next state of the environment, the reward, if the game is done, and additional information.
        # Modify the line where env.step() is called
        step_result = env.step(action)
        # print("Step result:", step_result)
        # Then modify the line where the unpacking occurs
        # Then modify the line where the unpacking occurs
        n_state, reward, done, _, info = step_result

        # n_state, reward, done, info = env.step(action)

        # Note: other env functions
        # env.reset() - Reset the environment and obtain initial observations.
        # env.render() - Render the environment. Visualise the environment.
        # env.step() - Step though the environment. Take an action and return the next state, the reward, if the game is done, and additional information.
        # env.close() - Close the environment / the render frame.

        score += reward
    print("Episode:{} Score:{}".format(episode, score))

env.close()
# env.render(close=True)
# display_frames_as_gif(frames)
