# Advantage Actor Critic Model (A2C)

> Actor-Critic model trained using value advantages.  Environment is from the [OpenAI Gym CartPole-V0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) environment.

![Sample Output](/images/A2C_CartPole_TrainingStats.png)

Environment can be changed to other OpenAI environments that have a discrete output space in the main method.

## Python Libraries

This script was built using the following python libraries and versions:

* python v3.7.6
* numpy v1.16.5
* matplotlib v3.1.1
* keras 2.3.1
* gym 0.15.3

## References

1. Rokas Balsys; Reinforcement Learning tutorial; [https://pylessons.com/A2C-reinforcement-learning/](https://pylessons.com/A2C-reinforcement-learning/)
2. Actor Critic with OpenAI Gym; Quadcopter Dynamics and Simulation;
    [http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html](http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html)