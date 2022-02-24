# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:16:04 2022

@author: Da Huo
"""
import numpy as np
import math
import random
import pandas as pd
import multiprocessing as mp
import gym
import gym_openpit
from gym_openpit.envs.openpit_locations import OpenpitLocations, Locations_enum

class simulate_openpit():
    def __init__(self, NUM_AGENTS, discount_factor=0.6, DECAY_FACTOR=800, MIN_EXPLORE_RATE=0.05, MIN_LEARNING_RATE=0.4):
        
        self.env = gym.make("openpit-v0")
        self.NUM_AGENTS = NUM_AGENTS
        self.num_periods = 41000#num_periods#2100
        self.run_ID = 's5b' + str(self.NUM_AGENTS) #run_ID
        self.discount_factor = discount_factor
        
        self.output_path = '/Users/Desktop/GitProjects/outputs/'#output_path
        self.env.init_env(self.NUM_AGENTS, self.run_ID, self.num_periods, self.output_path)
        self.obs = self.env.reset()
        self.q_table = self.env.fill_state_space()
        
        self.DECAY_FACTOR = DECAY_FACTOR
        self.MIN_EXPLORE_RATE = MIN_EXPLORE_RATE
        self.MIN_LEARNING_RATE = MIN_LEARNING_RATE
        
    def simulate(self):       
        # Instantiating the learning related parameters
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        # Render tha maze
        #env.render()
        state_0 = self.env.reset()
        total_reward = 0
        t = 0
             
        while True:
            t += 1
            action_n = []
      
            for i in range(self.NUM_AGENTS):
                location = self.env.getRobotLocation(i)
                action_n.append(self.select_action(state_0[i], explore_rate, location, i))            
       
            state_n, reward_n, done_n, agent_status = self.env.step(action_n)

            for i in range(self.NUM_AGENTS):
                if agent_status[i] != 'enroute' and agent_status[i] != 'in mill queue' and agent_status[i] != 'in shovel queue' :
                    state = state_n[i]
                    action = action_n[i]
                    reward = reward_n[i]
                    total_reward += np.sum(reward_n[i])
                    
                    #print(state_0[i])                
                    # Update the Q based on the result
                    best_q = np.amax(self.q_table[i][state])
                    self.q_table[i][tuple(state_0[i]) + (action,)] += learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[i][tuple(state_0[i]) + (action,)])
                    
                    # Setting up for the next iteration
                    state_0[i] = state
                            
            if done_n[0]:# and done_n[1] and done_n[2]:
                self.env.render(close=True)
                self.env.close() 
                break
    
            explore_rate = self.get_explore_rate(t)
            learning_rate = self.get_learning_rate(t)
    
    def select_action(self, state, explore_rate, location, agent_ID):
        # Select a random action
        if random.random() < explore_rate:
            action = int(self.env.action_space.sample(location))
    
        # Select the action with the highest q
        else:
            action = int(np.argmax(self.q_table[agent_ID][tuple(state)]))
        return action    
    
    def get_explore_rate(self, t, setting = 'med'):
        if setting == 'med':
            return max(self.MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))
    
    def get_learning_rate(self, t):
        return max(self.MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))
        #return 0.8

    # fixed truck-shovel baseline 
    def simulate_baseline(self):
        #env.render()
        baseline_actions = pd.read_excel('C:/Users/Desktop/GitProjects/baseline_actions2.xlsx')
        total_reward = 0
        t = 0
        step_pointer = np.zeros(self.NUM_AGENTS, dtype=int)
        n = 100#204#baseline_actions['Agent0_Schedule2'].size
        
        while True:
            action_n = []
            for i in range(self.NUM_AGENTS):
                if i%3 == 0:
                    action_n.append(Locations_enum[baseline_actions['Agent0_Schedule2'][step_pointer[i]%n]]-1)
                elif i%3 == 1:
                    action_n.append(Locations_enum[baseline_actions['Agent1_Schedule2'][step_pointer[i]%n]]-1)
                else:
                    action_n.append(Locations_enum[baseline_actions['Agent2_Schedule2'][step_pointer[i]%n]]-1)
                    
            state_n, reward_n, done_n, agent_status = self.env.step(action_n)
            total_reward += np.sum(reward_n)
                 
            for i in range(self.NUM_AGENTS):
                if agent_status[i] != 'enroute' and agent_status[i] != 'in mill queue' and agent_status[i] != 'in shovel queue' :    
                    step_pointer[i] += 1
            
            t += 1
            if done_n[0]:# and done_n[1] and done_n[2]:
                self.env.render(close=True)
                self.env.close() 
                break
            
            
def start_simulations(NUM_AGENTS):
    simulation = simulate_openpit(NUM_AGENTS)
    simulation.simulate()
    print('Completed simulation for fleet size =', NUM_AGENTS)
    
def start_simulations_baseline(NUM_AGENTS):
    simulation = simulate_openpit(NUM_AGENTS)
    simulation.simulate_baseline()
    print('Completed simulation for fleet size =', NUM_AGENTS)
    
    
'''
    
def simulate(NUM_AGENTS):       
    #env = gym.make("openpit-v0")
    NUM_AGENTS = NUM_AGENTS
    num_periods = 2100
    run_ID = 'a' + str(NUM_AGENTS) #run_ID
    
    output_path = '/Users/Desktop/GitProjects/'
    env.init_env(NUM_AGENTS, run_ID, num_periods, output_path)
    obs = env.reset()
    q_table = env.fill_state_space()
    
    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    # Render tha maze
    #env.render()
    state_0 = env.reset()
    total_reward = 0
    t = 0
         
    while True:
        t += 1
        action_n = []
  
        for i in range(NUM_AGENTS):
            location = env.getRobotLocation(i)
            action_n.append(select_action(state_0[i], explore_rate, location, i))            
   
        state_n, reward_n, done_n, agent_status = env.step(action_n)

        for i in range(NUM_AGENTS):
            if agent_status[i] != 'enroute' and agent_status[i] != 'in mill queue' and agent_status[i] != 'in shovel queue' :
                state = state_n[i]
                action = action_n[i]
                reward = reward_n[i]
                total_reward += np.sum(reward_n[i])
                
                #print(state_0[i])                
                # Update the Q based on the result
                best_q = np.amax(q_table[i][state])
                q_table[i][tuple(state_0[i]) + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[i][tuple(state_0[i]) + (action,)])
                
                # Setting up for the next iteration
                state_0[i] = state
                        
        if done_n[0]:# and done_n[1] and done_n[2]:
            env.render(close=True)
            env.close() 
            break

        explore_rate = get_explore_rate(t)
        learning_rate = get_learning_rate(t)

def select_action(state, explore_rate, location, agent_ID):
    # Select a random action
    if random.random() < explore_rate:
        action = int(env.action_space.sample(location))

    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[agent_ID][tuple(state)]))
    return action    

def get_explore_rate(t, setting = 'med'):
    if setting == 'med':
        return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))
    #return 0.8
'''
if __name__ == "__main__":

    # for NUM_AGENTS in range(11,13):
    #     run_ID = 'a' + str(NUM_AGENTS)
    #     simulation = simulate_openpit(NUM_AGENTS, output_path, num_periods, run_ID)
    #     simulation.simulate()

    ##======== parallelize loop ======== 
    
    # obj_list = [simulate_openpit(i) for i in range(1,4)]
    # a_pool = mp.Pool()
    # a_pool.map(lambda simulation: simulation.simulate(), obj_list)
    
    a_pool = mp.Pool()  
    #a_pool.map(start_simulations, range(15,16))
    ### baseline simulation:
    a_pool.map(start_simulations_baseline, range(15,16)) #range(15,21,5)  
    #start_simulations_baseline(3)

        

        