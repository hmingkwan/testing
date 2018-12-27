# Continuous Control
***************************
### Project details
***************************
This project uses Reacher Unity environment. In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of our agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Environment
***************************
The environment contains 20 double joined arms agents which could move freely to reach the target locations.

### Goal
***************************
The environment is considered solved when the agent reaches average score of 30.0 over 100 consecutive episodes.

### Getting started
***************************
1. Follow the [instructions](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) to install Unity ML-Agents.

2. Download the environment from one of the links below. You need only select the environment that matches your operating system:
* Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

The project mainly uses Jupyter Notebook. Follow the instructions in `Continuous_Control.ipynb` to install dependencies and get started with training your own agent!

### Solution
***************************
Run `Continuous_Control.ipynb` for further details.
