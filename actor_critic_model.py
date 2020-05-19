import numpy as np 
import matplotlib.pyplot as plt

import os
import random
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
        self.goal_position = 0.55
        self.episode = 0
        self.episode_rewards = []
        self.episode_wins = []
        self.aggregate_episode_rewards = {'episode': [], 'avg_wins': [], 'avg_rewards': [], 'max_rewards': [], 'min_rewards': []}

        self.aggregate_stats_interval = 10
        self.show_stats_interval = self.aggregate_stats_interval
        self.save_models_interval = 10

        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]

        self.max_episodes = 500
        
        self.gamma = 0.99 #discount rate for rewards
        self.learning_rate = 0.001
        self.gradient_clipValue = 1.0

        self.epsilon = 1
        self.min_epsilon = 0.1
        self.episode_start_epsilon_decay = 1
        self.episode_end_epsilon_decay = self.max_episodes
        self.epsilon_decay_value = self.epsilon/(self.episode_end_epsilon_decay - self.episode_start_epsilon_decay)
        
        self.actorFC1size = 128
        self.actorFC2size = 64

        self.criticStateFC1size = 128
        self.criticStateFC2size = 64

        self.batchSize = 40
        self.buffer = 80
        
        #Bucket observations space into 20 bins.
        bins = 20
        Discrete_obs_size = [bins] * len(env.observation_space.high)
        self.discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/Discrete_obs_size
        #print(discrete_os_win_size)
        
        self.memory = []

        # make actor and critic networks
        self.actor_state_input, self.actor = self.create_actor_network()
        self.critic_state_input, self.critic = self.create_critic_network()

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
        actor_optimizer = Adam(lr = self.learning_rate, clipvalue = self.gradient_clipValue)
        model.compile(optimizer = actor_optimizer, loss = 'categorical_crossentropy')
        return state_input, model


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


    def save_models(self, path):
        self.actor.save(os.path.join(path, self.model_name + '_actor.h5'))
        self.critic.save(os.path.join(path, self.model_name + '_critic.h5'))


    def load_models(self, path):
        self.actor.load(os.path.join(path, self.model_name + '_actor.h5'))
        self.critic.load(os.path.join(path, self.model_name + '_critic.h5'))


    def remember(self, current_discrete_state, current_value, current_reward, action, new_discrete_state, new_value, new_reward, done):
        ''' Store episode states, actions, and rewards to memory'''
        self.memory.append([current_discrete_state, current_value, current_reward, action, new_discrete_state, new_value, new_reward, done])


    def discount_reward(self, org_reward, new_reward):
        ''' reduce the value of future rewards '''
        discounted_reward = org_reward + self.gamma * new_reward
        return discounted_reward


    def train_critic(self):
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

    def train_actor(self):
        ''' update the actor model based on the learned new values for states '''
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

                q_val = old_qval
                q_val[0,action] = actor_delta

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
        return action


    def value(self, state):
        ''' Determine the value of the current state with the critic '''
        value = self.critic.predict(state.reshape(1,self.state_size))
        return value


    def get_discrete_state(self, state):
        ''' returns the discrete bin that the current state is in '''
        discrete_state = (state - self.env.observation_space.low) / self.discrete_os_win_size
        return discrete_state.astype(np.int)


    def end_of_episode_actions(self, cumulative_reward):
        ''' Update epsilon, gather episode stats, show stats, and update episode number '''
        if (self.epsilon > self.min_epsilon) and (self.episode_end_epsilon_decay > self.episode > self.episode_start_epsilon_decay):
            self.epsilon -= self.epsilon_decay_value
        
        self.episode_rewards.append(cumulative_reward)

        ending_state = self.memory[-1][4]
        ending_y_state = ending_state[0]
        goal_y_state = self.get_discrete_state(self.goal_position)[0]
        if ending_y_state >= goal_y_state:
            self.episode_wins.append(1)
        else:
            self.episode_wins.append(0)

        if (not self.episode % self.aggregate_stats_interval) or (self.episode == self.max_episodes):
            # Find average, min, and max rewards over teh aggregate window
            reward_set = self.episode_rewards[-self.aggregate_stats_interval:]
            win_set = self.episode_wins[-self.aggregate_stats_interval:]
            average_reward = sum(reward_set)/len(reward_set)
            average_wins = sum(win_set)/len(win_set)
            min_reward = min(reward_set)
            max_reward = max(reward_set)
            self.aggregate_episode_rewards['episode'].append(self.episode)
            self.aggregate_episode_rewards['avg_wins'].append(average_wins)
            self.aggregate_episode_rewards['avg_rewards'].append(average_reward)
            self.aggregate_episode_rewards['min_rewards'].append(min_reward)
            self.aggregate_episode_rewards['max_rewards'].append(max_reward)

        if (not self.episode % self.show_stats_interval) or (self.episode == self.max_episodes):
            print(f"Episode: {self.episode}, average wins: {average_wins:.1f}, average reward: {average_reward:.2f}, current epsilon: {self.epsilon:.2f}")

        if (not self.episode % self.save_models_interval) or (self.episode == self.max_episodes):
            self.save_models(self.save_path)
        
        self.episode += 1


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

            current_state = self.env.reset()
            current_discrete_state = self.get_discrete_state(current_state)
            current_reward = 0
            cumulative_reward = 0
            done = False

            while not done:
                # Get the critic's value of the state
                current_value = self.value(current_discrete_state)

                # Get that action from the actor based on the state
                action = self.action(current_discrete_state)

                #Take action, observe new state S'
                new_state, new_reward, done, info = self.env.step(action)
                new_discrete_state = self.get_discrete_state(new_state)

                # Track eposide rewards
                cumulative_reward += new_reward

                #Get the critic's value of the new state S'
                new_value = self.value(new_discrete_state)

                # Remember states and values
                self.remember(current_discrete_state, current_value, current_reward, action, new_discrete_state, new_value, new_reward, done)

                # Call critic update
                self.train_critic()

                # call actor update
                self.train_actor()

                # End of turn actions
                current_state = new_discrete_state
                current_reward = new_reward
            
            # End of episode actions
            self.end_of_episode_actions(cumulative_reward)


def main():
    env = gym.make("MountainCar-v0")
    
    #Actions space: 0 = left, 1 = stay, 2 = right
    print(env.action_space.n)
    print(env.action_space)
    #How large is the observation space? - (x-pos, vel)   
    print(env.observation_space)    
    print(env.observation_space.high)
    print(env.observation_space.low)
    
    # Create the actor-critic model
    actor_critic = ActorCriticAgent(env)

    # Train the actor-critic model in the specified environment
    actor_critic.run()


    # visualize completed training run
    fig = plt.figure(figsize=(15,15))
    fig.tight_layout(pad=0.5)
    plt.style.use('fivethirtyeight')

    plt.title('Actor Critic Training Progress')
    plt.plot(actor_critic.aggregate_episode_rewards['episode'], actor_critic.aggregate_episode_rewards['avg_rewards'], label="average rewards")
    plt.plot(actor_critic.aggregate_episode_rewards['episode'], actor_critic.aggregate_episode_rewards['max_rewards'], label="max rewards")
    plt.plot(actor_critic.aggregate_episode_rewards['episode'], actor_critic.aggregate_episode_rewards['min_rewards'], label="min rewards")
    plt.legend(loc=2)
    plt.grid(True)
    plt.xlabel('Episodes')
    plt.ylabel('reward')
    plt.show()

    # watch the completed model in the environment
    current_state = env.reset()
    done = False
    while not done:
        current_discrete_state = actor_critic.get_discrete_state(current_state)
        action = actor_critic.action(current_discrete_state)
        new_state, reward_out, done, _ = env.step(action)
        env.render()      
        
    #env.close()


if __name__ == "__main__":
    main()












