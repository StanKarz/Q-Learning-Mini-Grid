import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np
from agent import QLearningAgent

# Hyper parameters

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005
max_steps = 50
discount_factor = 0.90
learning_rate = 0.1
total_episodes = 4000

env = ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0"))

agent = QLearningAgent(env, discount_factor=discount_factor, learning_rate=learning_rate, epsilon=max_epsilon)
rewards = np.zeros(total_episodes)


def to_tuple(array):
    try:
        return tuple(to_tuple(item) for item in array)
    except TypeError:
        return array


for episode in range(total_episodes):
    state = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        state = to_tuple(state)

        for i in range(0, 3):
            state_action = state, i
            if state_action not in agent.q_table:
                agent.q_table[state_action] = 0

        action = agent.take_action(state)

        next_state, reward, done, info = env.step(action)
        next_state = to_tuple(next_state)

        for i in range(0, 3):
            state_action = next_state, i
            if state_action not in agent.q_table:
                agent.q_table[state_action] = 0

        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_rewards += reward

        agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        if done:
            break
    rewards[episode] = total_rewards

print(agent.q_table.values())
