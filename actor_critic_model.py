import numpy as np 
import matplotlib.pyplot as plt

import os
import random
import math
from pathlib import Path

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam 

#Saving models requires the h5py lib as well.

import gym
from gym.envs.registration import register, spec


class ActorCriticAgent:
    ''' Creates the actor-critic RL agent '''

    def __init__(self, env):
        
        self.env = env
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        
        self.episode = 0
        self.episode_rewards = []
        self.aggregate_episode_rewards = {'episode': [], 'avg_rewards': [], 'max_rewards': [], 'min_rewards': []}

        self.aggregate_stats_interval = 10
        self.show_stats_interval = self.aggregate_stats_interval
        self.save_models_interval = 10

        self.max_episodes = 1000
        
        self.gamma = 0.95 #discount rate for rewards
        self.learning_rate = 0.0001

        self.epsilon = 1
        self.min_epsilon = 0.05
        self.episode_start_epsilon_decay = math.floor(0.25 * self.max_episodes)
        self.episode_end_epsilon_decay = math.floor(0.75 * self.max_episodes)
        self.epsilon_decay_value = self.epsilon/(self.episode_end_epsilon_decay - self.episode_start_epsilon_decay)
        
        self.actorFC1size = 128
        self.actorFC2size = 64

        self.criticStateFC1size = 128
        self.criticStateFC2size = 64
               
        self.memory = []

        # make actor and critic networks
        self.actor = self.create_actor_network()
        self.critic = self.create_critic_network()

        # create path to save model
        self.model_name = '{}_ActorCritic_{}'.format(self.env.unwrapped.spec.id, self.learning_rate)
        self.save_path = os.path.join(Path(__file__).parent, 'Models', self.model_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    
    def create_actor_network(self):
        ''' Create the actor (policy) network for the Actor-Critic Agent '''
        state_input = Input(shape = (self.state_size,) )
        actor_h1 = Dense(self.actorFC1size, activation = 'relu')(state_input)
        actor_h2 = Dense(self.actorFC2size, activation = 'relu')(actor_h1)
        actor_output = Dense(self.action_size, activation = 'softmax')(actor_h2)

        model = Model(inputs = state_input, outputs = actor_output)
        actor_optimizer = Adam(lr = self.learning_rate)
        model.compile(optimizer = actor_optimizer, loss = 'categorical_crossentropy')
        
        return model


    def create_critic_network(self):
        ''' Create the critic (value) network for the A2C Agent '''      

        state_input = Input(shape = (self.state_size,) )
        critic_h1 = Dense(self.criticStateFC1size, activation = 'relu')(state_input)
        critic_h2 = Dense(self.criticStateFC2size, activation = 'relu')(critic_h1)
        critic_output = Dense(1, activation = 'linear')(critic_h2)

        model = Model(inputs = state_input, outputs = critic_output)
        critic_optimizer = Adam(lr = self.learning_rate)
        model.compile(optimizer = critic_optimizer, loss = 'mse')
        
        return model


    def save_models(self, path):
        self.actor.save(os.path.join(path, self.model_name + '_actor.h5'))
        self.critic.save(os.path.join(path, self.model_name + '_critic.h5'))


    def load_models(self, path):
        self.actor.load(os.path.join(path, self.model_name + '_actor.h5'))
        self.critic.load(os.path.join(path, self.model_name + '_critic.h5'))


    def remember(self, state, action, reward, done):
        ''' Convert actions to onehot vectors and store episode states, one hot actions, and rewards to memory'''
        # Convert actions to one hot encoded vectors
        action_onehot = np.zeros(self.action_size)
        action_onehot[action] = 1

        # Store state, action vectors, rewards
        self.memory.append([state, action_onehot, reward, done])


    def discount_reward(self, reward):
        ''' reduce the value of future rewards '''
        Qval = 0
        discounted_reward = np.zeros_like(reward)
        
        for i in reversed(range(len(reward))):
            Qval = (Qval * self.gamma) + reward[i]
            discounted_reward[i] = Qval
        
        return discounted_reward


    def train_models(self):
        ''' update the critic model based on the learned new values for states '''
        # Extract experiences from memory
        states_from_memory = []
        actions_from_memory = []
        rewards_from_memory = []
        for item in self.memory:
            states_from_memory.append(item[0])
            actions_from_memory.append(item[1])
            rewards_from_memory.append(item[2])

        # Reshape for training
        states = np.vstack(states_from_memory)
        actions = np.vstack(actions_from_memory)

        # Compute discounted reward
        discounted_rewards = self.discount_reward(rewards_from_memory)
        discounted_rewards -= np.mean(discounted_rewards) # normalizing the result
        discounted_rewards /= np.std(discounted_rewards) # divide by standard deviation

        # Get critic predictions
        values = self.critic.predict(states)[:,0]

        # Compute advantages (delta of value from predicted value)
        advantages = discounted_rewards - values

        # Train the actor and critic
        self.actor.fit(states, actions, sample_weight = advantages, epochs = 1, verbose = 0)
        self.critic.fit(states, discounted_rewards, epochs = 1, verbose = 0) 

        
    def action(self, state, train):
        '''Predict the next action to take with the actor network using the current policy'''
        # Random draw vs. epsilon value. This encourages exploration.  Over time, epsilon will decrease to
        # encourage expotation. 
        if (random.random() < self.epsilon) and train:
            # Random draw loss so choose a random action
            action = np.random.choice(np.arange(self.action_size))
        else:
            # Use the actor to predict the action probabilities 
            # with the current state and then flatten to a single vector.
            action_probabilities = self.actor.predict(state.reshape(1,self.state_size))
            action = (np.argmax(action_probabilities))
        return action


    def end_of_episode_actions(self, cumulative_reward):
        ''' Update epsilon, reset memory, gather episode stats, show stats, and update episode number '''
        # increase episode counter and remember cumulative reward
        self.episode += 1
        self.episode_rewards.append(cumulative_reward)

        # decrease epsilon (probability of random action)
        if (self.epsilon > self.min_epsilon) and (self.episode_end_epsilon_decay > self.episode > self.episode_start_epsilon_decay):
            self.epsilon -= self.epsilon_decay_value
        
        # store aggregate results
        if (not self.episode % self.aggregate_stats_interval) or (self.episode == self.max_episodes):
            # Find average, min, and max rewards over teh aggregate window
            reward_set = self.episode_rewards[-self.aggregate_stats_interval:]
            average_reward = sum(reward_set)/len(reward_set)
            min_reward = min(reward_set)
            max_reward = max(reward_set)
            self.aggregate_episode_rewards['episode'].append(self.episode)
            self.aggregate_episode_rewards['avg_rewards'].append(average_reward)
            self.aggregate_episode_rewards['min_rewards'].append(min_reward)
            self.aggregate_episode_rewards['max_rewards'].append(max_reward)

        # show ongoing training stats
        if (not self.episode % self.show_stats_interval) or (self.episode == self.max_episodes):
            print(f"Episode: {self.episode}, average reward: {average_reward:.2f}, current epsilon: {self.epsilon:.2f}")

        # save models
        if (not self.episode % self.save_models_interval) or (self.episode == self.max_episodes):
            self.save_models(self.save_path)
        
        # Reset memory and increase episode counter
        self.memory = []


    def plot_training_results(self):
        fig = plt.figure(figsize=(8,8))
        fig.tight_layout(pad=0.5)
        plt.style.use('fivethirtyeight')

        plt.title('Actor Critic Training Progress')
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['avg_rewards'], label="average rewards")
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['max_rewards'], label="max rewards")
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['min_rewards'], label="min rewards")
        plt.legend(loc=2)
        plt.grid(True)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()


    def run(self):
        for episode in range(self.max_episodes):

            state = self.env.reset()
            #current_discrete_state = self.get_discrete_state(current_state)
            current_reward = 0
            cumulative_reward = 0
            done = False
            train = True
        
            while not done:
                # Get that action from the actor based on the state
                action = self.action(state, train)
                #Take action, observe new state S'
                new_state, reward, done, _ = self.env.step(action)
                # Remember states and values
                self.remember(state, action, reward, done)

                # End of turn actions
                state = new_state
                cumulative_reward += reward

            # Train actor and critic with experiences
            self.train_models()           
            # End of episode actions
            self.end_of_episode_actions(cumulative_reward)


def main():
    env_name = 'CartPole-v0'

    # Create the environment
    env = gym.make(env_name)
    
    # Create the actor-critic model
    actor_critic = ActorCriticAgent(env)

    # Train the actor-critic model in the specified environment
    actor_critic.run()

    # visualize completed training run
    actor_critic.plot_training_results()

    # watch the completed model in the environment
    current_state = env.reset()
    done = False
    train = False
    while not done:
        #current_discrete_state = actor_critic.get_discrete_state(current_state)
        action = actor_critic.action(current_state, train)
        current_state, reward_out, done, _ = env.step(action)
        env.render()      
        
    #env.close()


if __name__ == "__main__":
    main()












