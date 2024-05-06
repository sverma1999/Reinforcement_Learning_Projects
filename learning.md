# Reinforcement learning

Reinforcement learning focuses on teaching agents through trail and error.

## Four fundamental concepts that lay most RL projects are:

- Agent: the actor operating within the environment, it is usually governed by a policy (a rule that decides what action to take).

- Environment: the world in which the agent can operate in

- Action: the agent can do something within the environment known as an action.

- Reward and Observations: in return the agent receives a reward and a view of hat the environment looks like after acting on it.

## Applications of RL including:

- Autonomous Driving: Training a car to be able to navigate to the open world.
- Securities Trading: Train your agent to make trades that are going to you profit.
- Neural Network Architecture Search: Build and find optimal neural network for specific use-case.
- Simulated Training of Robots: Build simulated environment for that particular robot and train that robot to do particular thing.

## Limitations and Considerations

- For simple problems RL can be overkill
- Assumes the environment is Markovian.
  - Meaning, it assumes future states are depends on current observations, and not random acts.
    Example: A robot might not be able to act on, walking people next to it or someone touching it, because they are randoms acts.
- Training can take a long time and is not always stable.
  - Exploration: Agent explores the environment when start out.
  - Exploitation: Agent can exploit environment to get the best possible results.
    It can sometime have less time to explore and start exploitating it too early. Sometimes, we need to tune the hyperparameters, so out model can truly explore the environment and understand it.
    When we don't get that quite right, our model not all be that stable, so we might get to a certain point where we reach a cap in terms of maximum reward.


## Simulated vs Real Environments
- Real environments are often more complex and harder to model. Training in real environments can be dangerous and expensive.
- Simulated environments are often easier to model and can be trained in a safe and controlled environment. They can also be scaled up to train multiple agents at once.

### OpenAI Gym
- OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms.
- It provides you with an easy way to build environments for your agents to train in.
```python
# Stable Baselines Install:
https://www.gymlibrary.dev/
```

#### OpenAI Gym Spaces
This gives you a kicking off point rather than having to write all of the code yourself.
- Box: A n-dimensional box (tensor) that can take any range of values.
  - E.g. Box(0,1,shape=(3,3)) would be a 3x3 matrix of values between 0 and 1.
- Discrete: A fixed range of non-negative numbers.
  - E.g. Discrete(3) would be 0, 1, or 2.
- Tuple: A tuple of other spaces. (not supported by stable baselines 3)
  - E.g. Tuple(Discrete(3), Box(0,100,shape=(3,3))) would be a discrete value and a 3x3 matrix of values between 0 and 100.
- Dict: A dictionary of spaces.
  - E.g. Dict({'height': Discrete(2), 'speed': Box(0,100,shape=(3,3))}) would be a dictionary with a discrete value and a 3x3 matrix of values between 0 and 100.
- MultiBinary: A binary space of n bits. One hot encoded binary values.
  - E.g. MultiBinary(4) would be a 4-bit binary number.
- MultiDiscrete: A set of discrete values.
  - E.g. MultiDiscrete([3, 2, 2]) would be a set of 3 discrete values, the first with 3 options, the second with 2, and the third with 2.









## Learning:

1. All the basics to get up and started with Reinforcement Learning
2. How to build custom environments using OpenAI Gym
3. About working on custom projects for Reinforcement Learning

## Links Mentioned

Stable Baselines 3: https://stable-baselines3.readthedocs...
  - Stable Baselines 3 is a set of reliable implementations of reinforcement learning algorithms in PyTorch and TensorFlow.
OpenAI Gym: https://gym.openai.com/
PyTorch: https://pytorch.org/
Atarimania ROMs: http://www.atarimania.com/roms/Roms.rar
Swig: http://www.swig.org/Doc1.3/Windows.html



## Practical Tutorial:

- Create conda environment and activate it: `conda activate rl_pract`
- Create requirements.txt file and `stable-baselines3[extra]`
- Install the requirements: `pip install -r requirements.txt`




## Training Strategies

- Train for longer
- Hyperparameter tuning
- Try different algorithms

