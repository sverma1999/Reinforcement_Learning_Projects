# Reinforcement_Learning_Projects



## Project: Self-Driving Car
The goal of this project was to achieve an award score of 900+ for the agent, utilizing OpenAI's Gym environment and Reinforcement Learning (RL) algorithms. The project aimed to observe the effects of different strategies for RL and enable the agent to understand the environment around it, taking appropriate actions accordingly.


### Pre-requisites
- Install swig for `macOS`
    ```bash
    brew install swig
    ```
    - For `ubuntu`
    ```bash
    sudo apt-get install swig
    ```

### Project set up:
- Create conda environment and activate it
    ```bash
    conda create -n rl_pract python=3.9 -y
    ```
- Activate the environment
    ```bash
    conda activate rl_pract
    ```

- install all the requirements
    ```bash
    pip install -r requirements.txt
    ```
- Go to the `Self_Driving_Project` directory
    ```bash
    cd Self_Driving_Project
    ```

Note: Game is solved if the agent is able to get award score of 900+.
- If you want to only test the model, you can skip the training part and directly run the `Evaluation.py` file

- Run the `Training.py` file (Let it run for a while)
    ```bash
    python Training.py
    ```
    
    - This will train the model and save it in the `Training/Trained_Models/{model name}` directory
    - Logs will be saved in the `Logs` directory, you can run Tensorboard to see the training progress
    - My agent with 4 millions steps took around 9 hours to train on cloud GPU. Here is the machine's basic configuration:
        - GPU: 1 NVIDIA L4
        - CPU: 16 vCPUs
        - Memory: 24 GB
        
- There are 2 trained models by me (found inside the `Training/Trained_Models` directory):
    - Agent trained for 4,000,000 steps named `CarRacingPPO_4_million`
    - Agent trained for 50,000 steps named `CarRacingPPO_50k`

- Run the `Evaluation.py` file (Once the training is done)
    ```bash
    python Evaluation.py
    ```
- Check the pop-up window to see the agent in action

- Here is what I have observed:
    - On average, the agent gets the reward score of 900+, 2/5 times in the evaluation phase.
    - Many times, it stuck in one frame and not move
    - Other times, it will create a loop and keep on repeating the same actions.
- Some possible ways to improve the agent:
    - Train the agent for more steps
    - Change the hyperparameters
    - Change the algorithm
    - Change the reward function
    - Add more complex environment
    - Add more complex actions
    - Add more complex observations
    - Add more complex state space
    - Add more complex action space
    - Add more complex reward space

## Demo of the driving car agent trained for 50,000 steps vs 4,000,000 steps:
Checkout the demo at `Reinforcement_Learning_Practice/Self_Driving_Project/Video_Demo/Demo_50k_vs_4m_480p.mov`
