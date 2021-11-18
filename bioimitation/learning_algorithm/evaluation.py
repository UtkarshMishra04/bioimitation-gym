from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module,
             env: gym.Env,
             num_episodes: int) -> Dict[str, float]:

    stats = {'return': [], 'length': []}
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        total_reward = 0
        episode_length = 0
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

        stats['return'].append(total_reward)
        stats['length'].append(episode_length)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
