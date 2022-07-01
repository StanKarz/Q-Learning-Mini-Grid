import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from agent import QLearningAgent

env = ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0"))

# Hyper parameters
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0008
max_steps = 50
discount_factor = 0.95
learning_rate = 0.15
total_episodes = 2000

agent = QLearningAgent(env, discount_factor=discount_factor, learning_rate=learning_rate, epsilon=max_epsilon)
rewards = np.zeros(total_episodes)
rewards_dict = {}
rewards_array = []


def to_tuple(array):
    try:
        return tuple(to_tuple(item) for item in array)
    except TypeError:
        return array


for episode in range(0, total_episodes + 1):
    state = env.reset()
    total_rewards = 0
    actions_taken = []

    for step in range(max_steps):
        state = to_tuple(state)

        # Check if state action pair is in Q-table, if not assign a value of 0
        for i in range(0, 3):
            state_action = state, i
            if state_action not in agent.q_table:
                agent.q_table[state_action] = 0

        action = agent.take_action(state, step)
        actions_taken.append(action)

        next_state, reward, done, info = env.step(action)
        next_state = to_tuple(next_state)

        for i in range(0, 3):
            state_action = next_state, i
            if state_action not in agent.q_table:
                agent.q_table[state_action] = 0

        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_rewards += (reward + agent.get_noise(0, 0.05))

        # Decay epsilon exponentially 
        agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        if done:
            break

    # New dict containing total reward, episode and seed is added to a list at the end of each episode
    rewards_dict['total_reward'] = total_rewards
    rewards_dict['episode_number'] = episode
    rewards_dict['seed'] = 351
    rewards_dict['distribution_std'] = 0.05
    rewards_array.append(dict(rewards_dict))

q_table = agent.q_table
q_table_values = agent.q_table.values()
print(q_table)
print(q_table_values)