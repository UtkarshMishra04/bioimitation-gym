import random
import numpy as np
import opensim
import os
import math
from collections import deque
from bioimitation.imitation_envs.utils.opensim_utils import read_from_storage
from bioimitation.imitation_envs.utils.opensim_utils import kinematics_MSE
from bioimitation.imitation_envs.utils.opensim_utils import body_kinematics_MSE
from bioimitation.imitation_envs.utils.opensim_utils import extract_data_frame_by_index
from bioimitation.imitation_envs.utils.opensim_utils import convert_to_relative
from bioimitation.imitation_envs.utils.opensim_utils import normalize_forces
from bioimitation.imitation_envs.utils.opensim_utils import construct_predictive_model
from bioimitation.imitation_envs.utils.opensim_environment import OsimEnv


class MuscleRunningImitationEnv2D(OsimEnv):
    def __init__(self,
                 config
                 ):

        if config['mode'] == 'test':
            self.test = True
        else:
            self.test = False

        visualize = config['visualize']
        self.max_actuation = config['max_actuation']
        self.log = config['log']
        self.w_imitate = config['r_weights'][0]
        self.w_action = config['r_weights'][2]
        self.curr_action = None
        self.last_action = None
        self.max_action_horizon = config['horizon']
        self.action_history = deque([], maxlen=self.max_action_horizon)
        self.old_pos_pelvisx = 0
        self.use_target_obs = config['use_target_obs']
        self.use_GRF = config['use_GRF']

        step_size = 0.01
        integration_accuracy = 1e-3
        subject_dir = os.path.abspath('./bioimitation/imitation_envs/data/2D/')
        model_input_file = os.path.join(subject_dir, 'scale/model_scaled.osim')
        model_file = os.path.join(subject_dir, 'scale/model_predictive.osim')
        q_desired_file = os.path.join(subject_dir,
                            'running_reference_data/task_Kinematics_q.sto')
        u_desired_file = os.path.join(subject_dir,
                            'running_reference_data/task_Kinematics_u.sto')
        body_pos_desired_file = os.path.join(
                            subject_dir, 'running_reference_data/task_BodyKinematics_pos_global.sto')
        body_vel_desired_file = os.path.join(
                            subject_dir, 'running_reference_data/task_BodyKinematics_vel_global.sto')

        if os.path.isfile(model_file):
            pass #os.remove(model_file)
            # construct_predictive_model(model_input_file, model_file)
        else:
            construct_predictive_model(model_input_file, model_file)

        self.q_d = read_from_storage(model_file, q_desired_file, step_size)
        self.u_d = read_from_storage(model_file, u_desired_file, step_size)

        self.a_d = self.u_d.diff() / step_size  # numerical differentiation of velocity
        self.a_d.time = self.u_d.time  # restore time because of diff
        self.a_d.iloc[0] = self.a_d.iloc[1]  # mirror first point to avoid NaN

        self.x_d = read_from_storage(model_file, body_pos_desired_file,
                                     step_size)
        self.v_d = read_from_storage(model_file, body_vel_desired_file,
                                     step_size)

        if self.test:
            self.N = (self.q_d.shape[0] - 2)  # N is used to terminate the simulation
        else:
            self.N = self.q_d.shape[0] - 2  # Calculated steps for completion of one gait cycle

        super(MuscleRunningImitationEnv2D,
              self).__init__(model_file, step_size, integration_accuracy,
                             visualize)

        if config['apply_perturbations']:
            self.prescribed_force = opensim.PrescribedForce()
            self.prescribed_force.setBodyName('/bodyset/torso')
            self.prescribed_force.setPointFunctions(opensim.Constant(0),
                                                    opensim.Constant(0),
                                                    opensim.Constant(0))
            fx_force = opensim.PiecewiseConstantFunction()
            t_force = np.linspace(0, 10, 100, endpoint=True)
            for i in range(t_force.shape[0]):
                if np.fmod(t_force[i],2) > 1.8 :
                    force = np.random.choice([-50,50])
                    fx_force.addPoint(t_force[i], float(force))
                else:
                    fx_force.addPoint(t_force[i], 0)

            self.prescribed_force.setForceFunctions(fx_force, opensim.Constant(0), opensim.Constant(0))
            self.osim_model.model.addForce(self.prescribed_force)
            self.osim_model.model.initSystem()

    def get_mass(self):
        self.mass = self.osim_model.model.getTotalMass(self.osim_model.state)
        return self.mass

    def get_height(self):
        # height should be changed according to the model to be used
        self.height = 1.80
        return self.height

    def get_gravity(self):
        self.gravity = self.osim_model.model.getGravity()
        return self.gravity

    def step(self, action, obs_as_dict=False):
        '''Redefined because we want obs_as_dict=False by default. Also,
        implement a PD control law.'''

        if np.any(np.isnan(action)):
            # raise ValueError('NaN passed in the activation vector. ')
            action = np.zeros(action.shape)

        if self.last_action is None:
            self.last_action = action
            for iters in range(self.max_action_horizon):
                self.action_history.append(np.array(action))

        self.action_history.append(np.array(action))
        self.curr_action = np.mean(self.action_history, axis=0) #action

        return super(MuscleRunningImitationEnv2D, self).step(action, obs_as_dict)

    def reset(self, obs_as_dict=False):
        '''Random initial state from desired values.'''
        self.osim_model.reset()
        self.state_cache = None
        self.last_action = None

        # initial state is randomly initialized from desired values

        if self.test:
            index = 0
        else:
            index = random.randint(0, self.N/2)  # it seems with this it is easier to learn

        self.osim_model.set_time(self.q_d.iloc[index]['time'])
        self.osim_model.set_coordinates(
            extract_data_frame_by_index(self.q_d, index))
        self.osim_model.set_velocities(
            extract_data_frame_by_index(self.u_d, index))

        # return observation
        if obs_as_dict:
            return self.get_observation_dict()
        else:
            return np.array(self.get_observation())

    def get_state_dict(self):
        '''We want to simplify this agent so the observation space is
        smaller.'''
        self.state_dict = {}
        self.state_dict.update(self.osim_model.calc_joint_kinematics())
        self.state_dict.update(self.osim_model.calc_body_kinematics())
        self.state_dict.update(self.osim_model.calc_muscles_info())
        self.state_dict.update(self.osim_model.calc_forces_info())

        state_dict = self.state_dict.copy()

        index = self.osim_model.istep

        observation = {}

        # The estimate index of completeion of one gait cycle was calculated
        # from the healthy gait data and is used to calculate the current phase
        # of the simulation This is found useful to learn periodic gaits easily
        # from only one step by learning the gait cycle

        self.cycle = 70  # Calculated steps for completion of one gait cycle

        observation['phase'] = self.osim_model.istep / self.cycle - math.floor(
            self.osim_model.istep / self.cycle)

        observation['coordinate_pos'] = state_dict['coordinate_pos']
        observation['coordinate_vel'] = state_dict['coordinate_vel']
        observation['coordinate_acc'] = state_dict['coordinate_acc']

        if self.use_target_obs:
            observation['target_coordinate_pos'] = self.q_d.iloc[index+1].drop(['time','pelvis_tx']).to_dict()
            observation['target_coordinate_vel'] = self.u_d.iloc[index+1].drop(['time','pelvis_tx']).to_dict()

        observation['body_pos'] = {}
        observation['body_pos']['torso'] = state_dict['body_pos']['torso']
        observation['body_pos']['calcn_r'] = state_dict['body_pos']['calcn_r']
        observation['body_pos']['calcn_l'] = state_dict['body_pos']['calcn_l']
        observation['body_pos']['femur_r'] = state_dict['body_pos']['femur_r']
        observation['body_pos']['femur_l'] = state_dict['body_pos']['femur_l']
        observation['body_pos']['tibia_r'] = state_dict['body_pos']['tibia_r']
        observation['body_pos']['tibia_l'] = state_dict['body_pos']['tibia_l']
        observation['body_pos']['talus_r'] = state_dict['body_pos']['talus_r']
        observation['body_pos']['talus_l'] = state_dict['body_pos']['talus_l']
        observation['body_pos']['center_of_mass'] = state_dict['body_pos'][
            'center_of_mass']

        observation['body_vel'] = {}
        observation['body_vel']['torso'] = state_dict['body_vel']['torso']
        observation['body_vel']['calcn_r'] = state_dict['body_vel']['calcn_r']
        observation['body_vel']['calcn_l'] = state_dict['body_vel']['calcn_l']
        observation['body_vel']['center_of_mass'] = state_dict['body_vel'][
            'center_of_mass']

        observation['muscles'] = {}
        for muscle in self.osim_model.muscle_names:
            observation['muscles'][muscle] = {}
            observation['muscles'][muscle]['activation'] = state_dict['muscles'][muscle]['activation']
            observation['muscles'][muscle]['fiber_length'] = state_dict['muscles'][muscle]['fiber_length']
            observation['muscles'][muscle]['fiber_velocity'] = state_dict['muscles'][muscle]['fiber_velocity']

        observation = convert_to_relative(observation.copy())

        if self.use_GRF:
            observation['contact_forces'] = state_dict['contact_forces']
            

            observation = normalize_forces(observation,self.get_mass(),self.get_gravity(),self.get_height())

        self.pelvis_pos_x = observation['coordinate_pos']['pelvis_tx']
        self.pelvis_pos_y = observation['coordinate_pos']['pelvis_ty']

        del observation['coordinate_pos']['pelvis_tx']
        del observation['coordinate_pos']['pelvis_ty']

        return observation

    def get_limit_forces(self):
        # Dimitar: maybe include this in observation
        return list(self.osim_model.calc_forces_info()
                    ['coordinate_limit_forces'].values())

    def is_done(self):
        '''Simulation is done if:

        1. torso.y < 0.75

        2. |max coordinate limiting forces| > 1000 to prevent slow down
        of integrator

        3. |max acceleration| > 100000

        4. istep >= N (the demonstration steps)

        '''
        observations = self.state_dict.copy()
        torso_pos = observations['body_pos']['torso']
        force_limit = self.get_limit_forces()
        acceleration = list(observations['coordinate_acc'].values())
        terminal_height = 0.75

        if torso_pos[1] < terminal_height:
            return True
        elif np.max(np.abs(force_limit)) > 10000:
            return True
        elif np.max(np.abs(acceleration)) > 1000000:
            return True
        elif self.osim_model.istep >= self.N:
            return True
        else:
            return False

    def get_reward(self):
        w_c = 1
        w_v = 0.1
        w_a = 1
        w_l = 1
        w_e = 1
        w_o = 1
        w_f = 1

        s_c = 30
        s_v = 0.1
        s_a = 0.001
        s_l = 0.01
        s_e = 0.1
        s_o = 20
        s_f = 20

        observations = self.state_dict

        observations['coordinate_pos']['pelvis_tx'] = self.pelvis_pos_x
        observations['coordinate_pos']['pelvis_ty'] = self.pelvis_pos_y

        index = self.osim_model.istep

        q_error = kinematics_MSE(observations, self.q_d, index,
                                 'coordinate_pos')
        u_error = kinematics_MSE(observations, self.u_d, index,
                                 'coordinate_vel')

        CoM_error = body_kinematics_MSE(observations, self.x_d, index,
                                        'body_pos', 'center_of_mass')

        femur_r_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'femur_r')
        femur_l_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'femur_l')

        tibia_r_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'tibia_r')
        tibia_l_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'tibia_l')

        talus_r_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'talus_r')
        talus_l_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'talus_l')

        calcn_r_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'calcn_r')
        calcn_l_error = body_kinematics_MSE(observations, self.x_d, index,
                                            'body_pos', 'calcn_l')

        CoM_error = body_kinematics_MSE(observations, self.x_d, index,
                                        'body_pos', 'center_of_mass')

        position_r = w_c * np.exp(-s_c * q_error)
        com_r = w_o * np.exp(-s_o * CoM_error)
        foot_r = w_f / 2.0 * np.exp(
            -s_f *
            (calcn_r_error + femur_r_error + tibia_r_error + talus_r_error))
        foot_l = w_f / 2.0 * np.exp(
            -s_f *
            (calcn_l_error + femur_l_error + tibia_l_error + talus_l_error))

        activations = []
        for muscle in self.osim_model.muscle_names:
            activations.append(observations['muscles'][muscle]['activation'])
        a_error = np.exp(-2*np.linalg.norm(activations))
        effort = self.calc_cost_of_transport()/(20*np.array(activations).shape[0]**2)
        effort_r = np.exp(-effort/max((self.pelvis_pos_x - self.old_pos_pelvisx + 1),1))
        action_r = np.exp(-np.linalg.norm(self.curr_action - self.last_action))

        

        reward = (0.5+self.w_imitate)*(position_r * com_r) + self.w_effort*effort_r \
            + self.w_action*action_r

        self.last_action = self.curr_action
        self.old_pos_pelvisx = self.pelvis_pos_x

        if self.log:
            print('####################################')
            print(f'position     = {position_r:.5f}')
            print(f'CoM          = {com_r:.5f}')
            print(f'foot_r       = {foot_r:.5f}')
            print(f'foot_l       = {foot_l:.5f}')
            print(f'effort_r     = {effort_r:.5f}')
            print(f'action_r     = {action_r:.5f}')
            print(f'reward       = {reward:.5f}')
            print('####################################')

        return reward, [position_r, com_r, foot_l, foot_r, a_error]

    def calc_cost_of_transport(self):

        effort_init = 1.51 * self.osim_model.model.getTotalMass(self.osim_model.state)

        specific_tension = 0.25e6
        muscle_density = 1059.7

        total_effort = effort_init

        slowTwitchFiberRatios = [0.499, 0.55, 0.5, 0.484, 0.546, 0.759, 0.721, 0.499, 0.55, 0.5, 0.484, 0.546, 0.759, 0.721]

        for i in range(self.osim_model.model.getMuscles().getSize()):

            muscle = self.osim_model.model.getMuscles().get(i)

            mus_mass = (muscle.getMaxIsometricForce()/specific_tension) * muscle_density * muscle.getOptimalFiberLength()

            l = slowTwitchFiberRatios[i]

            fa = 40*l*math.sin(0.5*np.pi*muscle.getExcitation(self.osim_model.state)) \
                + 133*(1-l)*(1-math.cos(0.5*np.pi*muscle.getExcitation(self.osim_model.state)))
            fm = 74*l*math.sin(0.5*np.pi*muscle.getActivation(self.osim_model.state)) \
                + 111*(1-l)*(1-math.cos(0.5*np.pi*muscle.getActivation(self.osim_model.state)))
            l_ce_norm = muscle.getFiberLength(self.osim_model.state) / muscle.getOptimalFiberLength()
            v_ce = muscle.getFiberVelocity(self.osim_model.state)

            g = 0.0

            if l_ce_norm < 0.5:
                g = 0.5
            elif l_ce_norm < 1.0:
                g = l_ce_norm
            elif l_ce_norm < 1.5:
                g = -2 * l_ce_norm + 3

            effort_a = mus_mass * fa
            effort_m = mus_mass * g * fm
            effort_s = max( 0.0, 0.25 * muscle.getFiberForce(self.osim_model.state) * -v_ce )
            effort_w = max( 0.0, muscle.getActiveFiberForce(self.osim_model.state) * -v_ce )
            effort = effort_a + effort_m + effort_s + effort_w

            total_effort += effort

        return total_effort