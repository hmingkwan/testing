# Continuous Control Project Report
***************************
### Algorithm
***************************
DDPG agent algorithm is used in this project.

```sh
The network comprises of 2 networks:
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
The agents were able to solve task in 85 episodes with a final average score of 20.36. 
![alt](https://github.com/hmingkwan/Projects/blob/master/continuous_control/images/result.png)

### Ideas for future work
***************************
* Build an agent that finds the best hyperparameters for an agent
* Prioritization for replay buffer
* Paramter space noise for better exploration
* Implement PPO, D3PG or D4PG that might produce better results
* Test separate replay buffer for agents
