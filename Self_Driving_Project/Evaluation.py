import os
import gym
from stable_baselines3 import PPO

# from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_model(model, env, episodes):
    # Evaluate the model
    # evaluate_policy(model, env, n_eval_episodes=episodes, render=True)
    # env.close()

    # or
    episodes = 5
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action, _states = model.predict(obs)
            obs, reward, _, done, info = env.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))
    env.close()


if __name__ == "__main__":
    # Test the Environment setup
    environment_name = "CarRacing-v2"

    # Keep this render_mode="human" to see the environment
    env = gym.make(environment_name, render_mode="human")
    # env = gym.make(environment_name)

    model_name = "CarRacingPPO_temp"
    trained_model_path = os.path.join("Training", "Trained_Models", model_name)

    # Evaluate the model
    # trained_model_path = os.path.join(
    #     "Training", "Trained_Models", "PPO_2m_Driving_model_2"
    # )

    print("Model Evaluation Started....")

    model = PPO.load(trained_model_path)
    evaluate_model(model, env, episodes=5)

    print("Model Evaluation Completed")
    print("+++++++++++++++++++++++++++++++++++++++++++++")
