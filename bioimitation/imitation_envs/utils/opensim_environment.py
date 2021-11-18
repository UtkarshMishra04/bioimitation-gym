import gym
import copy
import numpy as np
from gym import spaces
from flatten_dict import flatten
from bioimitation.imitation_envs.utils.opensim_wrapper import OsimModel


class Specification:
    timestep_limit = None
    def __init__(self, timestep_limit):
        self.id = 0
        self.timestep_limit = timestep_limit


class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 300


def convert_to_gym(space):
    return spaces.Box(np.array(space[0]), np.array(space[1]))


def gymify_env(env):
    env.action_space = convert_to_gym(env.action_space)
    env.observation_space = convert_to_gym(env.observation_space)
    env.spec = Specification(env.timestep_limit)
    env.spec.action_space = env.action_space
    env.spec.observation_space = env.observation_space
    return env


class OsimEnv(gym.Env):
    def __init__(self, model_path, step_size, integrator_accuracy, visualize):
        self.osim_model = OsimModel(model_path, step_size, integrator_accuracy,
                                    visualize)
        self.state_cache = None  # state chache to improve efficiency
        self.osim_model.reset()

        # create specs, action and observation spaces mocks for compatibility
        # with OpenAI gym
        self.action_space = (self.osim_model.action_min,
                             self.osim_model.action_max)
        self.observation_space = ([-np.inf] * self.get_observation_space_size(),
                                  [np.inf] * self.get_observation_space_size())
        self.timestep_limit = 1e10
        
        gymify_env(self)

    def get_observation(self):
        observation = []
        observation_dict = flatten(self.get_observation_dict())
        for value in observation_dict.values():
            if isinstance(value, float):
                observation.append(value)
            elif isinstance(value, list):
                for v in value:
                    observation.append(v)
            else:
                raise NotImplementedError

        return observation

    def get_observation_space_size(self):
        return len(self.get_observation())

    def get_action_space_size(self):
        return self.osim_model.get_action_space_size()

    def render(self, mode='human', close=False):
        return

    def get_reward(self):
        raise NotImplementedError('Should return two things.')

    def get_state_dict(self):
        raise NotImplementedError('Should be derived in base clase and include '
                                  'only the required calculations.')

    def is_done(self):
        return False

    def get_observation_dict(self):
        return self.get_state_dict()

    def reset(self, obs_as_dict=True):
        self.osim_model.reset()
        self.state_cache = None

        if obs_as_dict:
            return self.get_observation_dict()
        else:
            # Remember to change self.get_observation() to array if facing error
            # obs = np.array(self.get_observation())
            obs = self.get_observation()
            return np.array(obs)

    def step(self, action, obs_as_dict=True):
        self.osim_model.actuate(action)
        self.osim_model.integrate()
        self.state_cache = None  # invalidate state cache to recompute

        if obs_as_dict:
            obs = self.get_observation_dict()
        else:
            obs = self.get_observation()

        reward, all_rewards = self.get_reward()
        done = self.is_done()
        
        return [ np.array(obs), reward, done, {'all_rewards': all_rewards} ]
