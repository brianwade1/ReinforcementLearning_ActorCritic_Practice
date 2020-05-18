import numpy as np 

import os
import random

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam 

#Saving models requires the h5py lib as well.

from collections import deque

# Import custom libraries in current folder
#from Actor import Actor
#from Critic import Critic

import gym
from gym.envs.registration import register, spec



def to_onehot(size, value):
    my_onehot = np.zeros((size))
    my_onehot[value] = 1
    return my_onehot


class ActorCriticAgent:
    ''' Creates the actor-critic RL agent '''

    def __init__(self, env):
        
        self.env = env
        self.goal_position = 0.55
        self.episode = 0

        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]

        self.max_episodes = 50
        
        self.gamma = 0.99 #discount rate for rewards
        self.learning_rate = 0.001
        self.gradient_clipValue = 1.0

        self.epsilon = 1
        self.min_epsilon = 0.1
        self.episode_start_epsilon_decay = 1
        self.episode_end_epsilon_decay = self.max_episodes
        self.epsilon_decay_value = self.epsilon/(self.episode_end_epsilon_decay - self.episode_start_epsilon_decay)
        
        self.actorFC1size = 264
        self.actorFC2size = 128

        self.criticStateFC1size=264
        self.criticStateFC2size=128

        self.batchSize = 40
        self.buffer = 80
        
        #Bucket observations space into 20 bins.
        bins = 20
        Discrete_obs_size = [bins] * len(env.observation_space.high)
        self.discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/Discrete_obs_size
        #print(discrete_os_win_size)
        
        #self.memory = deque(maxlen=2000)
        self.memory = []

        # make actor and critic networks
        #self.Actor = Actor(self.action_size, self.state_size, self.learning_rate)
        #self.Critic = Critic(self.state_size, self.learning_rate)
        self.actor_state_input, self.actor = self.create_actor_network()
        self.critic_state_input, self.critic = self.create_critic_network()

        # create path to save model
        self.save_path = 'Models'
        self.model_name = '{}_ActorCritic_{}'.format(self.env.unwrapped.spec.id, self.learning_rate)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    
    def create_actor_network(self):
        ''' Create the actor (policy) network for the Actor-Critic Agent '''
        state_input = Input(shape = (self.state_size,) )
        actor_h1 = Dense(self.actorFC1size, activation = 'relu')(state_input)
        actor_h2 = Dense(self.actorFC2size, activation = 'relu')(actor_h1)
        actor_output = Dense(self.action_size, activation = 'softmax')(actor_h2)

        model = Model(inputs = state_input, outputs = actor_output)
        actor_optimizer = Adam(lr = self.learning_rate, clipvalue = self.gradient_clipValue)
        model.compile(optimizer = actor_optimizer, loss = 'categorical_crossentropy')
        return state_input, model

    
    def save_actor(self, path):
        self.model.save(path + '_actor.h5')


    def load_actor(self, path):
        self.model.load(path + '_actor.h5')

    
    def create_critic_network(self):
        ''' Create the critic (value) network for the A2C Agent '''      

        state_input = Input(shape = (self.state_size,) )
        critic_h1 = Dense(self.criticStateFC1size, activation = 'relu')(state_input)
        critic_h2 = Dense(self.criticStateFC2size, activation = 'relu')(critic_h1)
        critic_output = Dense(1, activation = 'linear')(critic_h2)

        model = Model(inputs = state_input, outputs = critic_output)
        critic_optimizer = Adam(lr = self.learning_rate, clipvalue = self.gradient_clipValue)
        model.compile(optimizer = critic_optimizer, loss = 'mse')
        return state_input, model


    def save_critic(self, path):
        self.model.save(path + '_critic.h5')


    def load_critic(self, path):
        self.model.load(path + '_critic.h5')


    def remember(self, current_discrete_state, current_value, current_reward, action, new_discrete_state, new_value, new_reward, done):
        ''' Store episode states, actions, and rewards to memory'''
        # self.states.append(state)
        # self.actions.append(action)
        # self.rewards.append(reward)
        # self.done.append(done)
        self.memory.append([current_discrete_state, current_value, current_reward, action, new_discrete_state, new_value, new_reward, done])


    def discount_reward(self, org_reward, new_reward):
        ''' reduce the value of future rewards '''
        discounted_reward = org_reward + self.gamma * new_reward
        return discounted_reward


    def _train_critic(self):
        ''' update the critic model based on the learned new values for states '''

        # Train if there are enough samples 
        if (len(self.memory)>= self.buffer):
            critic_replay = self.memory[-self.buffer:-1]
            samples = random.sample(critic_replay, self.batchSize)
            x = []
            y = []
            for sample in samples:
                current_state, current_value, current_reward, new_value, new_reward, done = [sample[i] for i in (0,1,2,5,6,7)]

                # If not in terminal state, discount the new value
                if not done:
                    target = self.discount_reward(current_reward, new_value)
                else:
                    # If in terminal state, the value comes directly from the environment (reward)
                    target = self.discount_reward(current_reward, new_reward)

                # When updating the critic, we use the best value for the state which is
                # the discounted rewards if the agent selects the best possible moves from 
                # this state forward. Additionally, add a small discount the value of the critic
                # network in case it is returning unusually large values.  As gamma decreases, 
                # these will be reduced to reasonable values.
                best_val = max((current_value * self.gamma), target)

                # Store the current states and the best value of each sample in the x,y for training the 
                #critic network.
                #x.append(current_state.reshape((self.state_size[0],)))
                x.append(current_state)
                y.append(np.array(best_val).reshape(1,))
                

            # Train the critic network
            x = np.array(x)
            y = np.array(y)
            self.critic.fit(x, y, batch_size = self.batchSize, epochs = 1, verbose = 0)

    def _train_actor(self):

        # Train if there are enough samples 
        if (len(self.memory)>= self.buffer):
            actor_replay = self.memory[-self.buffer:-1]
            samples = random.sample(actor_replay, self.batchSize)
            x = []
            y = []
   
            for sample in samples:
                current_state, current_value, action, new_value = [sample[i] for i in (0,1,3,5)]

                # The original q-value (probability of taking each of the available actions while in the state)
                old_qval = self.actor.predict(current_state.reshape(1,self.state_size))

                # Build the update for the Actor. The actor is updated
                # by using the difference of the value the critic
                # placed on the old state vs. the value the critic
                # places on the new state. encouraging the actor
                # to move into more valuable states.
                actor_delta = new_value - current_value
                
                #q_val = np.zeros((1, self.action_size))
                #q_val[:] = old_qval[:]

                q_val = old_qval
                q_val[0,action] = actor_delta

                #q_val = actor_delta

                #x.append(current_state.reshape((self.state_size,)))
                x.append(current_state)
                y.append(q_val.reshape(self.action_size,))
            
            #Train the actor model
            x = np.array(x)
            y = np.array(y)
            self.actor.fit(x, y, batch_size = self.batchSize, epochs = 1, verbose = 0)


    def action(self, state):
        '''Predict the next action to take with the actor network using the current policy'''
        # Random draw vs. epsilon value. This encourages exploration.  Over time, epsilon will decrease to
        # encourage expotation. 
        if (random.random() < self.epsilon):
            # Random draw loss so choose a random action
            action = np.random.choice(np.arange(self.action_size))
        else:
            # Use the actor to predict the action probabilities 
            # with the current state and then flatten to a single vector.
            action_probabilities = self.actor.predict(state.reshape(1,self.state_size))
            action = (np.argmax(action_probabilities))

        #action_probabilities = self.actor.predict(state).reshape(-1)
        #action = np.random.choice(np.arange(self.action_size), 1, p = action_probabilities)[0]
        
        #action_onehot = np.zeros([self.action_size])
        #action_onehot[action] = 1
        #return action_onehot.reshape(-1)
        return action

    def value(self, state):
        ''' Determine the value of the current state with the critic '''
        #value = self.critic.predict(np.array(state).reshape(1,2))
        value = self.critic.predict(state.reshape(1,self.state_size))
        #value = self.critic.predict(list(state))
        return value

    def end_of_episode_actions(self):
        ''' Update epsilon and episode number '''
        if (self.epsilon > self.min_epsilon) & (self.episode_end_epsilon_decay > self.episode > self.episode_start_epsilon_decay):
            self.epsilon -= self.epsilon_decay_value
        self.episode += 1

    def get_discrete_state(self, state):
        ''' returns the discrete bin that the current state is in '''

        discrete_state = (state - self.env.observation_space.low) / self.discrete_os_win_size
        #return tuple(discrete_state.astype(np.int))
        return discrete_state.astype(np.int)

    #def get_reward(self,state):
    #    if state[0] >= self.goal_position:
    #        #print("Car has reached the goal")
    #        return 10
    #    if state[0] > -0.4:
    #        return (1+state[0])**2
    #    return 0

    def run(self):
        for episode in range(self.max_episodes):

            print(f"starting episode number {episode + 1}")

            done = False
            current_state = self.env.reset()
            current_discrete_state = self.get_discrete_state(current_state)
            current_reward = 0

            while not done:
                # Get the critic's value of the state
                current_value = self.value(current_discrete_state)

                # Get that action from the actor based on the state
                action = self.action(current_discrete_state)

                #Take action, observe new state S'
                new_state, new_reward, done, info = self.env.step(action)
                new_discrete_state = self.get_discrete_state(new_state)

                #Get the critic's value of the new state S'
                new_value = self.value(new_discrete_state)

                # Remember states and values
                self.remember(current_discrete_state, current_value, current_reward, action, new_discrete_state, new_value, new_reward, done)

                # Call critic update
                self._train_critic()

                # call actor update
                self._train_actor()

                # End of turn actions
                current_state = new_discrete_state
                current_reward = new_reward
                self.end_of_episode_actions()

def main():
    env = gym.make("MountainCar-v0")
    
    #Actions space: 0 = left, 1 = stay, 2 = right
    print(env.action_space.n)
    print(env.action_space)
    #How large is the observation space? - (x-pos, vel)   
    print(env.observation_space)    
    print(env.observation_space.high)
    print(env.observation_space.low)
    
    actor_critic = ActorCriticAgent(env)

    actor_critic.run()


    # visualize completed training run
    
    current_state = env.reset()
    done = False
    while not done:

        current_discrete_state = actor_critic.get_discrete_state(current_state)
        action = actor_critic.action(current_discrete_state)
        new_state, reward_out, done, _ = env.step(action)
        env.render()      
        
    env.close()

if __name__ == "__main__":
    main()












