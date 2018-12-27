# Banana Navigation Project Report
***************************
### Algorithm
***************************
The learning algorithm is a Deep Q Learning as described in original paper. However, deep neural network is used. The deep neural network has 3 fully connected layers as following:

```sh
Fully connected layer - input: 37 (state size) output: 128
Fully connected layer - input: 128 output 64
Fully connected layer - input: 64 output: (action size)
```

The hyperparameters as followings:
```sh
Parameters used in DQN algorithm:
Maximum steps per episode: 1000
Starting epsilion: 1.0
Ending epsilion: 0.0001
Epsilion decay rate: 0.01
```
### Result
***************************
![image](https://github.com/hmingkwan/Projects/blob/master/banana_navigation/images/result.png)

### Ideas for future work
***************************
* Hyperparameter optimization
* Implement Double Deep Q Networks
* Implement Dueling Deep Q Networks
* Learning from pixels
