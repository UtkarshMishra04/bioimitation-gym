###########################################################################

# Utkarsh Mishra (umishra@me.iitr.ac.in)

# This is a sample file demonstrating usage of the imitation environment
# with different algorithms other than stable-baselines (main implementation)

# The gym register command can be used to generate the environment id to use
# for further training purposes

###########################################################################

import argparse
import numpy as np
import gym
import json
import ray
from absl import app, flags
from ml_collections import config_flags
import ray.rllib.agents.ppo as ppo
import bioimitation

parser = argparse.ArgumentParser()

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'test_config',
    'configs/test_default.py',
    'File path to the testing hyperparameter configuration.',
    lock_config=False)

def main(_):

    test_config = dict(FLAGS.test_config)

    args = parser.parse_args()
    ray.init()

    to_save = test_config["save_plots"]

    config = json.load(open(test_config["agentfile"]+'/params.json',))

    agent = ppo.PPOTrainer(config=config)

    agent.restore(test_config["agentfile"]+'/checkpoint/checkpoint-'+str(test_config["checkpoint"]))

    config["env_config"]["mode"] = test_config["mode"]
    config["env_config"]["visualize"] = test_config["visualize"]
    config["env_config"]["apply_perturbations"] = test_config["apply_perturbations"]
    env = gym.make(config["env"], config=config["env_config"])

    # run until episode ends

    max_reward = 0

    for i in range(test_config["max_iterations"]):
        episode_reward = 0
        done = False
        obs = env.reset()
        num_steps = 0
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            num_steps += 1

        print("Episode: {} || Reward: {}".format(i, episode_reward))

        if to_save and episode_reward > max_reward:
            env.osim_model.save_simulation(test_config["agentfile"])
            max_reward = episode_reward

    if to_save:
        if config["env"].find('2D'):
            from bioimitation.imitation_envs.visualization.visualize_kinematics2D import visualize_performance
        else:
            from bioimitation.imitation_envs.visualization.visualize_kinematics3D import visualize_performance
        visualize_performance(test_config["agentfile"])

    env.close()

if __name__ == '__main__':
    app.run(main)