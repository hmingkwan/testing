# Collaboration and Competition
***************************
### Algorithm
***************************
Multi Agent Deep Deterministic Policy Grdients (MADDPG) algorithm has been used to complete the project as the action space is continuous. The algorithm consists of 2 seperate agents each with set of 2 neural networks - actor and critic.

### Network Architectures
***************************
Actor Network
* Batch normalization on input.
* A hidden layer with 256 units, relu activation.
* Second hidden layer with 128 units, relu activation.
* Fully connected layer with tanh activation.

Critic Network
* Batch normalization on input.
* A hidden layer with 256 units, relu activation.
* Second hidden layer with 128 units, relu activation.
* Fully connected layer with tanh activation.


### Result
***************************
The agents were able to solve task in 2873 episodes with a final average score of 0.506. 
![alt](https://github.com/hmingkwan/Projects/blob/master/tennis/images/result.png)

### Ideas for future work
***************************
* Build an agent that finds the best hyperparameters for an agent
* Prioritization for replay buffer
* Paramter space noise for better exploration
* Test shared network between agents
* Test separate replay buffer for agents
