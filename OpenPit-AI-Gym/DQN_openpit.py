# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:46:46 2021

@author: Da Huo
"""
import gym
import sys
import random
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym_openpit
from keras.utils import to_categorical
#from gym_openpit.envs.openpit_locations import Locations_enum
#import os
#import segmentation_models as sm


class DQN_openpit():
    def __init__(self, state_size, action_size, NUM_AGENTS, EPISODES = 150, discount_factor=0.3):
        self.state_size = state_size
        self.action_size = action_size
        self.EPISODES = EPISODES
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 1e-3
        self.model = self.build_neural_net()

        self.env = gym.make("openpit-v0")
        self.NUM_AGENTS = NUM_AGENTS
        self.num_periods = 2000#num_periods#2100
        self.run_ID = 'DQN_a1' + str(self.NUM_AGENTS) #run_ID
        self.discount_factor = discount_factor
        
        self.output_path = '/Users/danie/Desktop/ComputeCanada/DQN_results/'#output_path
        self.env.init_env(self.NUM_AGENTS, self.run_ID, self.num_periods, self.output_path)
        self.obs = self.env.reset()       
                
    def build_neural_net(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='elu'))
        model.add(Dense(16, activation='elu'))
        model.add(Dense(16, activation='elu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, agent_ID, location):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            #return int(self.env.action_space.sample(location))
        #print(state)
        ##print(self.model.summary())
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
       
    # from RL book            
    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]   
        return states, actions, rewards, next_states, dones
    
    def replay2(self, batch_size):
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        loss_fn = tf.keras.losses.mean_squared_error
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                           (1 - dones) * self.discount_factor * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.action_size)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))    
    # end from RL book 

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def product(lst):
        p = 1
        for i in lst:
            p *= i
        return p
            
    def simulate(self):               
        #state_0 = env.reset()
        #state_size = env.observation_space.shape[0]
        #state_size = (env.reset()).shape
        #state_size = env.state_space_shape()
        #state_size = product(state_space)

        #action_size = self.env.action_space.n
        # agent.load("./save/cartpole-dqn.h5")
        done = False
        total_reward = 0
        batch_size = 256
        state_to_env = np.zeros((self.NUM_AGENTS,5), dtype=int).tolist()
        for e in range(self.EPISODES):
            #state = self.env.reset()        
            #state = np.reshape(state, [1, state_size])
            state = np.zeros((self.NUM_AGENTS,self.state_size), dtype=int).tolist()
            
            for time in range(50):
                # env.render()
                
                # multi-agent:
                action_n = []
                for i in range(self.NUM_AGENTS):
                    location = self.env.getRobotLocation(i)
                    #print(state[i])
                    #state[i] = np.reshape(state[i], [1, self.state_size])
                    state_to_env[i] = [np.argmax(state[i][0:self.action_size], axis=0)]+state[i][self.action_size:self.state_size]
                    action_n.append(self.select_action(state[i], i, location))
                
                # ============ Version: queue and checkpoints ===================
                # ===============================================================
                state_n, reward_n, done_n, agent_status = self.env.step(action_n)
                
                # single-self:
                # action = self.select_action(state)
                # next_state, reward, done, _ = self.env.step(action)
                
                for i in range(self.NUM_AGENTS):
                    if agent_status[i] != 'enroute' and agent_status[i] != 'in mill queue' and agent_status[i] != 'in shovel queue' :
                        next_state = tf.one_hot(state_n[i][0], self.action_size).numpy().tolist() + state_n[i][1:self.state_size]
                        # next_state = to_categorical(state_n[i])
                        action = action_n[i]
                        reward = reward_n[i]
                        total_reward += np.sum(reward_n[i])
        
                        #next_state = np.reshape(next_state, [1, self.state_size])
                        self.memorize(state, action, reward, next_state, done)
                        
                        # For the next iteration
                        state[i] = next_state
                        if len(self.memory) > batch_size:
                            self.replay(batch_size)
                            print("episode: {}/{}, action: {}, e: {:.2}"
                              .format(e, self.EPISODES, agent_status, self.epsilon))
                if done_n[0]:
                    print("episode: {}/{}, step: {}, e: {:.2}"
                          .format(e, self.EPISODES, time, self.epsilon))
                    self.env.render(close=True)
                    self.env.close()  
                    break    
            # if e % 10 == 0:
            #     self.save("./save/cartpole-dqn.h5")                
        print("Total reward = %f." % total_reward)            

if __name__ == "__main__":
    NUM_AGENTS = 1
    state_size = 5
    action_size = 9
    simulation = DQN_openpit(state_size, action_size, NUM_AGENTS)
    simulation.simulate()
    print('Completed simulation for fleet size =', NUM_AGENTS)
    
    
#%%============= previous working version ====================
if __name__ == "__main__":
    env = gym.make("openpit-v0")
    env.render()
    
    DEBUG_MODE = 0
    RENDER_MAZE = True
    ENABLE_RECORDING = True
    
    #state_0 = env.reset()
    #state_size = env.observation_space.shape[0]
    #state_size = (env.reset()).shape
    state_space = env.state_space_shape()
    def product(lst):
        p = 1
        for i in lst:
            p *= i
        return p
    #state_size = product(state_space)
    state_size = 5
    action_size = env.action_space.n
    agent = DQN_openpit(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    total_reward = 0
    batch_size = 256
    NUM_AGENTS = 1
    
    for e in range(EPISODES):
        state = env.reset()        
        #state = np.reshape(state, [1, state_size])        
        for time in range(50):
            # env.render()
            
            # multi-agent:
            action_n = []
            for i in range(NUM_AGENTS):
                action_n.append(agent.select_action(state[i], i))
            
            # ============ Version: queue and checkpoints ===================
            # ===============================================================
            state_n, reward_n, done_n, agent_status = env.step(action_n)
            
            # single-agent:
            # action = agent.select_action(state)
            # next_state, reward, done, _ = env.step(action)
            
            for i in range(NUM_AGENTS):
                if agent_status[i] != 'enroute' and agent_status[i] != 'in mill queue' and agent_status[i] != 'in shovel queue' :
                    next_state = state_n[i]
                    action = action_n[i]
                    reward = reward_n[i]
                    total_reward += np.sum(reward_n[i])
    
                    next_state = np.reshape(next_state, [1, state_size])
                    agent.memorize(state, action, reward, next_state, done)
                    
                    # For the next iteration
                    state[i] = next_state
                
                    # Render the layout
                    # if RENDER_MAZE:
                    #     env.render()    
                    # if env.is_game_over():
                    #     sys.exit()                
                    
                    # if done_n[0]:
                    #     print("episode: {}/{}, score: {}, e: {:.2}"
                    #           .format(e, EPISODES, time, agent.epsilon))
                    #     break
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                        print("episode: {}/{}, action: {}, e: {:.2}"
                          .format(e, EPISODES, agent_status, agent.epsilon))
            if done_n[0]:
                print("episode: {}/{}, step: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break    
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")                
    print("Total reward = %f." % total_reward)            
    env.render(close=True)
    env.close()                         
                

