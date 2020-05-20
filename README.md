# Advantage Actor Critic Model (A2C)

> Actor-Critic model trained using value advantages.  Environment is from the [OpenAI Gym CartPole-V0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) environment.

![Cartpole](/images/cartpole.png)

These scripts impliment an advantage actor critic (A2C) on the OpenAI cart pole environment. Sample training results are shown below. The environment can be changed to other OpenAI environments that have a discrete output space.

![Sample Output](/images/A2C_CartPole_TrainingStats.png)

## Python Libraries

This script was built using the following python libraries and versions:
* python v3.7.6
* numpy v1.16.5
* matplotlib v3.1.1
* keras 2.3.1
* gym 0.15.3

## References

1. Sutton, Richard S. and Barto, Andrew G. <ins>Reinforcement Learning: An Introduction</ins>, 2nd ed. The MIT Press; Cambridge, Massachusetts. 2018
2. Rokas Balsys; Reinforcement Learning tutorial; [https://pylessons.com/A2C-reinforcement-learning/](https://pylessons.com/A2C-reinforcement-learning/)
3. Actor Critic with OpenAI Gym; Quadcopter Dynamics and Simulation;
    [http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html](http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html)