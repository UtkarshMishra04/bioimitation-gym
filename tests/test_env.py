import os
from absl import app, flags
from ml_collections import config_flags
import gym
import bioimitation

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    'configs/env_default.py',
    'File path to the environment configuration.',
    lock_config=False)

def main(_):

    example_config = dict(FLAGS.config)

    env = gym.make("MuscleWalkingImitation2D-v0", config=example_config)

    env.reset()

    for i in range(1000):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()

if __name__ == '__main__':
    app.run(main)
