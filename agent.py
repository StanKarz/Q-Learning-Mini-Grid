import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import random
import numpy as np

env = ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0"))


class QLearningAgent:
    def __init__(self, env, epsilon, learning_rate, discount_factor):
        self.epsilon = epsilon
        self.gamma = discount_factor
        self.lr = learning_rate
        self.q_table = {}

    def take_action(self, state, step):
        # For exploration, select a relevant random action
        random.seed(351)
        random_actions_arr = [random.randint(0, 2) for i in range(0, step + 1)]
        random_action = random_actions_arr[step]
        # Generate random number from 0 to 1, to be used for comparison with epsilon value
        random_num = random.uniform(0, 1)

        # Retrieve max Q-value, given the current state
        max_q_val = max(self.q_table[state, 0], self.q_table[state, 1], self.q_table[state, 2])

        # Find the action associated with the highest Q-value
        for key, value in self.q_table.items():
            if key[0] == state and value == max_q_val:
                max_action = key[1]

        # Epsilon greedy policy
        if random_num > self.epsilon:
            action = max_action  # Action associated with max Q-value
        else:
            action = random_action
        return action

    def update_q_table(self, state, action, reward, next_state):
        # Store max Q-value associated with the next state
        max_q_val = max(self.q_table[next_state, 0], self.q_table[next_state, 1], self.q_table[next_state, 2])
        # Q-table update
        self.q_table[state, action] = self.q_table[state, action] + self.lr * (reward + (self.gamma * max_q_val) - self.q_table[state, action])

    def get_noise(self, mu, sigma):
        # mean and standard deviation
        distribution = np.random.normal(mu, sigma, 1000)
        noise = random.choice(distribution)
        return noise
