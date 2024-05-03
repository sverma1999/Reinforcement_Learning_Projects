import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


def environment_test(env):
    episodes = 5
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            # env.render() # but this slows down the training
            action = env.action_space.sample()
            obs, reward, _, done, info = env.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))
    env.close()


def train_PPO_model(policy_name, env, timesteps, log_path, model_path):
    # env = gym.make(environment_name)
    # env = VecFrameStack(env, n_stack=4)
    model = PPO(policy_name, env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    env.close()


def evaluate_model(model, env, episodes):
    # Evaluate the model
    evaluate_policy(model, env, n_eval_episodes=episodes, render=True)
    env.close()

    # or
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    # env.close()


if __name__ == "__main__":
    # Test the Environment setup
    environment_name = "CarRacing-v2"

    # Keep this render_mode="human" to see the environment
    # env = gym.make(environment_name, render_mode="human")
    env = gym.make(environment_name)

    environment_test(env)

    print("Environment Tested Successfully")
    print("+++++++++++++++++++++++++++++++++++++++++++++")

    # Train the model
    model_name = "CarRacingPPO"
    timesteps = 20000  # Ideally timesteps more than 2-5 million for better performance!
    policy_name = "CnnPolicy"
    trained_model_path = os.path.join("Training", "Trained_Models", model_name)
    log_path = os.path.join("Training", "Logs")

    train_PPO_model(
        policy_name=policy_name,
        env=env,
        timesteps=timesteps,
        log_path=log_path,
        model_path=trained_model_path,
    )

    print("Model Trained Successfully")
    print("+++++++++++++++++++++++++++++++++++++++++++++")

    # Evaluate the model
    model = PPO.load(trained_model_path, model_name)
    evaluate_model(model, env, episodes=5)

    print("Model Evaluation Completed")
    print("+++++++++++++++++++++++++++++++++++++++++++++")
