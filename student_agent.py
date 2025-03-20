# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
from torch import nn
from torch.distributions import Categorical

n_actions = 6
n_states = 7

class Policy_Network(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(Policy_Network, self).__init__()
        self.policy_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.tensor):
        return self.policy_layer(x)
    
class Value_Network(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Value_Network, self).__init__()
        self.value_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.tensor):
        return self.value_layer(x)

class Agent():
    def __init__(self):
        self.actor = Policy_Network(n_states, 512, n_actions)
        self.critic = Value_Network(n_states, 512)
        
        self.softmax = nn.Softmax(dim = -1)
        
    def choose_action(self, state):
        logit = self.actor(state)
        probs = self.softmax(logit)
        print(f"probs: {probs}")
        action = Categorical(probs).sample()
        return action.item()
        
    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location = torch.device("cpu"), weights_only = True))
        self.critic.load_state_dict(torch.load(critic_path, map_location = torch.device("cpu"), weights_only = True))

agent = Agent()
agent.load_model(f'actor.ckpt', f'critic.ckpt')

carrying = 0
stations_pos = None

desti_pos, passenger_pos = None, None
target = None

def get_target():
    global stations_pos, passenger_pos, desti_pos, carrying, target
    
    if carrying and desti_pos:
        return desti_pos
    elif not carrying and passenger_pos:
        return passenger_pos
    else:
        return stations_pos[0]
    
epsilon = 0.1

def get_action(obs):
    global stations_pos, passenger_pos, desti_pos, carrying, target, epsilon
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    taxi_pos = (obs[0], obs[1])
    if stations_pos == None:
        stations_pos = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        target = get_target()
    passenger_look = obs[14]
    destination_look = obs[15]
    
    dx = taxi_pos[0] - target[0]
    dx = 1 if dx > 0 else -1 if dx < 0 else 0
    dy = taxi_pos[1] - target[1]
    dy = 1 if dy > 0 else -1 if dy < 0 else 0

    state = torch.tensor([dx, dy, carrying, obs[10], obs[11], obs[12], obs[13]], dtype = torch.float32)
    action = agent.choose_action(state)
    
    # if np.random.rand() < epsilon:
    #     action = np.random.randint(n_actions)
    
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

