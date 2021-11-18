import argparse
import ray
from ray import tune
from ray.tune import grid_search
from absl import app, flags
from ml_collections import config_flags
import ray.rllib.agents.ppo as ppo
import bioimitation

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'MusclePalsyImitation3D-v0', 'Environment name.')
flags.DEFINE_string('output', './results/rllib/', 'Output logging dir.')
flags.DEFINE_integer('max_steps', int(2e7), 'Number of training steps.')

config_flags.DEFINE_config_file(
    'train_config',
    'configs/train_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

config_flags.DEFINE_config_file(
    'env_config',
    'configs/env_default.py',
    'File path to the Environment configuration.',
    lock_config=False)



def main(_):

    ray.init()

    env_name = FLAGS.env_name
    time_steps = FLAGS.max_steps
    
    output = FLAGS.output + env_name

    train_config = dict(FLAGS.train_config)     # Load training config  
    env_config = dict(FLAGS.env_config)         # Load environment config   

    train_config["env"] = env_name
    train_config["env_config"] = env_config

    stop = {
        "timesteps_total": time_steps,
    }

    for key in train_config.keys():
        if type(train_config[key]) is list:
            train_config[key] = grid_search(train_config[key])

    print("Training config:", train_config)

    tune.run(
        ppo.PPOTrainer, 
        config=train_config, 
        stop=stop, 
        local_dir=output, 
        checkpoint_freq=1,
        checkpoint_at_end=True
    )

    ray.shutdown()


if __name__ == '__main__':
    app.run(main)