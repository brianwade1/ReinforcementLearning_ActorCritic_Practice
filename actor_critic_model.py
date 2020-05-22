import numpy as np 
import matplotlib.pyplot as plt

import os
import random
import math
from datetime import datetime
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
        self.env_name = self.env.unwrapped.spec.id
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        
        self.episode = 0
        self.episode_rewards = []
        self.aggregate_episode_rewards = {'episode': [], 'avg_rewards': [], 'max_rewards': [], 'min_rewards': []}

        self.aggregate_stats_window = 10
        self.show_stats_interval = 100
        self.save_models_interval = 100

        self.save_training_plot = True

        self.max_episodes = 1000
        
        self.gamma = 0.95 #discount rate for rewards
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.005
        
        self.actorFC1size = 64
        self.actorFC2size = 24

        self.criticStateFC1size = 64
        self.criticStateFC2size = 24
               
        self.memory = []

        self.start_time = datetime.now()
        self.last_ep_start_time = datetime.now()

        # make actor and critic networks
        self.actor = self.create_actor_network()
        self.critic = self.create_critic_network()

        # create path to save model
        self.model_name = f'{self.env_name}_ActorCritic'
        self.save_path = os.path.join(Path(__file__).parent, 'Models', self.model_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # save parameters file for reference
        parameter_file = os.path.join(self.save_path, 'parameters.txt')
        with open(parameter_file, "w") as text_file:
            print(f"Environment: {self.env_name}\n", file = text_file)
            print(f"Future reward discount factor (gamma): {self.gamma}", file = text_file)
            print(f"Number of training episodes: {self.max_episodes}", file = text_file)
            print(f"Actor learning rate: {self.learning_rate_actor}", file = text_file)
            print(f"Critic learning rate: {self.learning_rate_actor}", file = text_file)
            print(f"Actor hiddel layer 1 size: {self.actorFC1size}", file = text_file)
            print(f"Actor hiddel layer 2 size: {self.actorFC2size}", file = text_file)
            print(f"Critic hiddel layer 1 size: {self.criticStateFC1size}", file = text_file)
            print(f"Critic hiddel layer 2 size: {self.criticStateFC2size}", file = text_file)

    
    def create_actor_network(self):
        ''' Create the actor (policy) network for the Actor-Critic Agent '''
        state_input = Input(shape = (self.state_size,) )
        actor_h1 = Dense(self.actorFC1size, activation = 'relu')(state_input)
        actor_h2 = Dense(self.actorFC2size, activation = 'relu')(actor_h1)
        actor_output = Dense(self.action_size, activation = 'softmax')(actor_h2)

        model = Model(inputs = state_input, outputs = actor_output)
        actor_optimizer = Adam(lr = self.learning_rate_actor)
        model.compile(optimizer = actor_optimizer, loss = 'categorical_crossentropy')
        
        return model


    def create_critic_network(self):
        ''' Create the critic (value) network for the A2C Agent '''      

        state_input = Input(shape = (self.state_size,) )
        critic_h1 = Dense(self.criticStateFC1size, activation = 'relu')(state_input)
        critic_h2 = Dense(self.criticStateFC2size, activation = 'relu')(critic_h1)
        critic_output = Dense(1, activation = 'linear')(critic_h2)

        model = Model(inputs = state_input, outputs = critic_output)
        critic_optimizer = Adam(lr = self.learning_rate_critic)
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
        #discounted_rewards -= np.mean(discounted_rewards) # normalizing the result
        #discounted_rewards /= np.std(discounted_rewards) # divide by standard deviation

        # Get critic predictions
        values = self.critic.predict(states)[:,0]

        # Compute advantages (delta of value from predicted value)
        advantages = discounted_rewards - values

        # Train the actor and critic
        self.actor.fit(states, actions, sample_weight = advantages, epochs = 1, verbose = 0)
        self.critic.fit(states, discounted_rewards, epochs = 1, verbose = 0) 

        
    def policy_action(self, state, train):
        '''Predict the next action to take with the actor network using the current policy'''

        action_policy = self.actor.predict(state.reshape(1,self.state_size)).ravel()
        if train:
            action = np.random.choice(self.action_size, 1, p=action_policy)[0]
        else:
            action = np.argmax(action_policy)
        
        return action


    def end_of_episode_actions(self, cumulative_reward):
        ''' Reset memory, gather episode stats, show stats, and update episode number '''
        # increase episode counter and remember cumulative reward
        self.episode += 1
        self.episode_rewards.append(cumulative_reward)
        
        # store aggregate results
        if (not self.episode % self.aggregate_stats_window) or (self.episode == self.max_episodes):
            # Find average, min, and max rewards over teh aggregate window
            reward_set = self.episode_rewards[-self.aggregate_stats_window:]
            average_reward = sum(reward_set)/len(reward_set)
            min_reward = min(reward_set)
            max_reward = max(reward_set)
            self.aggregate_episode_rewards['episode'].append(self.episode)
            self.aggregate_episode_rewards['avg_rewards'].append(average_reward)
            self.aggregate_episode_rewards['min_rewards'].append(min_reward)
            self.aggregate_episode_rewards['max_rewards'].append(max_reward)

        # show ongoing training stats
        if (not self.episode % self.show_stats_interval):
            # Find time for episode
            time_delta = datetime.now() - self.last_ep_start_time
            delta_min = (time_delta.seconds//60)
            delta_sec = (time_delta.seconds - delta_min * 60)%60
            print(f"Episode: {self.episode}, average reward: {average_reward:.2f}, time to complete: {delta_min} minutes, {delta_sec} seconds")
            self.last_ep_start_time = datetime.now()

        if (self.episode == self.max_episodes):
            time_delta = datetime.now() - self.start_time
            delta_hour = time_delta.seconds//3600
            delta_min = ((time_delta.seconds - (delta_hour * 3600))//60)
            delta_sec = (time_delta.seconds - delta_hour*3600 - delta_min * 60)%60
            print('--------- TRAINING COMPLETE -----------')
            print(f"Episode: {self.episode}, average reward: {average_reward:.2f}, time to complete: {delta_hour} hours, {delta_min} minutes, {delta_sec} seconds")

        # save models
        if (not self.episode % self.save_models_interval) or (self.episode == self.max_episodes):
            self.save_models(self.save_path)
        
        # Reset memory and increase episode counter
        self.memory = []


    def plot_training_results(self):
        fig = plt.figure(figsize=(8,8))
        fig.tight_layout(pad=0.5)
        plt.style.use('ggplot')

        plt.title('Actor Critic Training Progress')
        plt.plot(np.arange(1,len(self.episode_rewards)+1), self.episode_rewards, label='eposide rewards')
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['avg_rewards'], label="average rewards")
        #plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['max_rewards'], label="max rewards")
        #plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['min_rewards'], label="min rewards")
        plt.legend(loc=2)
        plt.grid(True)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()

        if self.save_training_plot:
            plot_name = f"training_progress_{self.model_name}.png"
            fig.savefig(os.path.join(self.save_path, plot_name))

    
    def visualize_episode(self):
        current_state = self.env.reset()
        done = False
        train = False
        while not done:
            #current_discrete_state = actor_critic.get_discrete_state(current_state)
            action = self.policy_action(current_state, train)
            current_state, reward_out, done, _ = self.env.step(action)
            self.env.render()


    def run(self):
        for episode in range(self.max_episodes):

            state = self.env.reset()
            #current_discrete_state = self.get_discrete_state(current_state)
            current_reward = 0
            cumulative_reward = 0
            done = False
            train = True
        
            while not done:
                # Get the action based on current policy from the actor
                action = self.policy_action(state, train)
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
    actor_critic.visualize_episode()


def play_trained_model(env, path):
    # initiate class
    actor_critic = ActorCriticAgent(env)
    # load trained actor and critic
    actor_critic.load_models(path)
    # watch the completed model in the environment
    actor_critic.visualize_episode()


if __name__ == "__main__":
    main()


