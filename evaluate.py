import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import main


env = ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0"))
eval_rewards_dict = {}
eval_rewards_array = []

# Hyper parameters
total_episodes = 500
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0008
max_steps = 50
discount_factor = 0.95
learning_rate = 0.15


def to_tuple(array):
    try:
        return tuple(to_tuple(item) for item in array)
    except TypeError:
        return array


for episode in range(total_episodes):
    state = env.reset()
    total_rewards = 0

    for step in range(30):
        state = to_tuple(state)

        # Retrieve max q-value, given the current state
        max_q_val = max(main.q_table[state, 0], main.q_table[state, 1], main.q_table[state, 2])

        # Find action associated with max q-value
        for key, value in main.q_table.items():
            if key[0] == state and value == max_q_val:
                max_action = key[1]

        action = max_action
        next_state, reward, done, info = env.step(action)
        next_state = to_tuple(next_state)
        state = next_state
        total_rewards += reward

        if done:
            break

    eval_rewards_dict['total_reward'] = total_rewards
    eval_rewards_dict['episode_number'] = episode
    eval_rewards_dict['seed'] = 351
    eval_rewards_dict['distribution_std'] = 0.05
    eval_rewards_array.append(dict(eval_rewards_dict))

print(eval_rewards_array)
