import os
import random
import bioimitation
import numpy as np
import bioimitation.learning_algorithm.agents
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import gym
from bioimitation.learning_algorithm.agents import SACLearner
from bioimitation.learning_algorithm.datasets import ReplayBuffer
from bioimitation.learning_algorithm.evaluation import evaluate

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'MuscleWalkingImitation2D-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './results/custom', 'Tensorboard logging dir.')
flags.DEFINE_integer('max_steps', int(2e7), 'Number of training steps.')

config_flags.DEFINE_config_file(
    'config_algo',
    'configs/baseline_train_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

config_flags.DEFINE_config_file(
    'env_config',
    'configs/env_default.py',
    'File path to the Environment configuration.',
    lock_config=False)

def main(_):

    seed = np.random.randint(2**10)

    kwargs_algo = dict(FLAGS.config_algo)
    kwargs_env = dict(FLAGS.env_config)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name)

    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', "seed-"+str(seed)))

    env = gym.make(FLAGS.env_name, config=kwargs_env)
    eval_env = gym.make(FLAGS.env_name, config=kwargs_env)

    np.random.seed(seed)
    random.seed(seed)

    replay_buffer_size = kwargs_algo.pop('replay_buffer_size')
    batch_size = kwargs_algo.pop('batch_size')
    eval_episodes = kwargs_algo.pop('eval_episodes')
    log_interval = kwargs_algo.pop('log_interval')
    eval_interval = kwargs_algo.pop('eval_interval')
    start_training = kwargs_algo.pop('start_training')
    
    agent = SACLearner(seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis], **kwargs_algo)


    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1):
        if i < start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        if i >= start_training:
            batch = replay_buffer.sample(batch_size)
            update_info = agent.update(batch)

            if i % log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
