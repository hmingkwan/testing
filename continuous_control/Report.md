# Continuous Control Project Report
***************************
### Learning Algorithm
***************************
DDPG agent algorithm, which is used in this project is based on the paper ["Continuous control with deep reinforcement learning"](https://arxiv.org/abs/1509.02971). This project is also an extension of the previous project - Banana Navigation in applying Deep Q-Network (DQN) to solve single agent navigation environment. However, this project has a more complex environment with continuous action spaces and multiple agents. 

### Network Architecture
```sh
The network consists of 2 networks:
Actor: 256 -> 256
Critic: 256 -> 256 -> 128

Hyperparameters:
replay buffer size = 1e6
minibatch size = 64
discount factor = 0.99
tau for soft update of target parameters = 1e-3
learning rate of the actor = 1e-4
learning rate of the critic = 3e-4
L2 weight decay = 0.0001
```
### Result
***************************
The agents were able to solve task in 185 episodes with a final average score of 20.36. 
* [version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.
![alt](https://github.com/hmingkwan/Projects/blob/master/continuous_control/images/result.png)

### Ideas for future work
***************************
* Build an agent that finds the best hyperparameters for an agent
* Prioritization for replay buffer
* Paramter space noise for better exploration
* Implement PPO, D3PG or D4PG that might produce better results
* Test separate replay buffer for agents
