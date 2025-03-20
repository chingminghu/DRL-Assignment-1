# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import pickle

from time import sleep # toDel

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()

n_actions = 6

carrying = 0
stations_pos = None

desti_pos, passenger_pos = None, None
target = None

def load_policy(filename="policy.pkl"):
    with open(filename, "rb") as f:
        policy_table = pickle.load(f)
    return policy_table

def get_target():
    global stations_pos, passenger_pos, desti_pos, carrying, target
    
    if carrying and desti_pos:
        return desti_pos
    elif not carrying and passenger_pos:
        return passenger_pos
    else:
        return stations_pos[0]
    
def get_state(obs):
    global stations_pos, passenger_pos, desti_pos, carrying, target
    
    taxi_pos = (obs[0], obs[1])
    if stations_pos is None:
        stations_pos = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        target = get_target()
    passenger_look = obs[14]
    destination_look = obs[15]
    
    dx = taxi_pos[0] - target[0]
    dx = 1 if dx > 0 else -1 if dx < 0 else 0
    dy = taxi_pos[1] - target[1]
    dy = 1 if dy > 0 else -1 if dy < 0 else 0

    state = (dx, dy, carrying, obs[10], obs[11], obs[12], obs[13])
    
    # print(f"state: {state}")
    
    return state

policy_table = load_policy()

def get_action(obs):
    global stations_pos, passenger_pos, desti_pos, carrying, target
    
    state = get_state(obs)
    
    if state not in policy_table:
        action = np.random.randint(n_actions)
    else:
        prob = softmax(policy_table[state])
        # print(f"probs: {prob}")
        action = np.random.choice(n_actions, p = prob)
    
    taxi_pos = (obs[0], obs[1])
    target = get_target()
    
    # print(f"target: {target}")
    
    passenger_look = obs[14]
    destination_look = obs[15]
    
    if stations_pos and taxi_pos in stations_pos:
        if passenger_look and passenger_pos is None:
            passenger_pos = taxi_pos
        elif destination_look and desti_pos is None:
            desti_pos = taxi_pos
        stations_pos.remove(taxi_pos)
        target = get_target()
    
    if carrying and action == 5:
        passenger_pos = taxi_pos
        carrying = 0
        target = get_target()
        
    elif not carrying and action == 4 and taxi_pos == passenger_pos:
        carrying = 1
        target = get_target()
    
    return action

