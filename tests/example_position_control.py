# A script that demonstrates the use of OpenSim's environment for solving torque
# position control.
#
# author: Dimitar Stanev (dimitar.stanev@epfl.ch)
# %%
import os
import opensim
import argparse
import numpy as np
from gym import spaces

from opensim_utils import convert_model_to_torque_actuated
from opensim_utils import read_from_storage
from opensim_utils import extract_data_frame_by_index
from opensim_environment import OsimEnv

import sys
sys.path.append('./inverse_dynamics/')
import inverse_dynamics

# %%
# custom environment


def convert_to_gym(space):
    return spaces.Box(np.array(space[0]), np.array(space[1]))


def index_correspondence(from_list, to_list):
    return [from_list.index(element) for element in to_list]


class PositionControlEnv(OsimEnv):
    def __init__(self, model_file, step_size, integration_accuracy, visualize,
                 q_desired_file, u_desired_file, max_actuation, use_id,
                 use_stable_pd):
        super(PositionControlEnv,
              self).__init__(model_file, step_size, integration_accuracy,
                             visualize)
        # make sure we have removed the muscles
        assert (self.osim_model.model.getMuscles().getSize() == 0)

        # load desired coordinates and speeds, and compute acceleration
        self.q_d = read_from_storage(model_file, q_desired_file, step_size)
        self.u_d = read_from_storage(model_file, u_desired_file, step_size)
        self.a_d = self.u_d.diff() / step_size  # numerical differentiation
        self.a_d.time = self.u_d.time  # restore time because of diff
        self.a_d.iloc[0] = self.a_d.iloc[1]  # mirror first point to avoid NaN

        # N is used to terminate the simulation; -2 because in PD we do t + 1
        self.N = self.q_d.shape[0] - 2

        # these might not be used here, but defined for safety
        self.max_actuation = max_actuation
        self.action_space = (self.osim_model.coordinate_limit_min,
                             self.osim_model.coordinate_limit_max)
        self.action_space = convert_to_gym(self.action_space)

        # used for computing the integral error
        self.integral_error = None

        # inverse dynamics
        self.use_id = use_id
        self.use_stable_pd = use_stable_pd
        if self.use_id or self.use_stable_pd:
            self.Kp = 2000
            self.Kd = 500
            self.Ki = 0
            self.id_model = inverse_dynamics.InverseDynamics(model_file)

            # calculate index mapping (due to > v3.3 of OpenSim)
            self.normal_to_multibody_order = index_correspondence(
                self.osim_model.get_coordinate_names(),
                self.osim_model.get_coordinate_names_multibody_order())
            self.multibody_to_normal_order = index_correspondence(
                self.osim_model.get_coordinate_names_multibody_order(),
                self.osim_model.get_coordinate_names())
        else:
            self.Kp = 2000
            self.Kd = 10
            self.Ki = 1

    def step(self, action, obs_as_dict=False):
        # get current state
        obs_dict = self.get_observation_dict()
        q = obs_dict['coordinate_pos']
        u = obs_dict['coordinate_vel']

        # get desired state at i + 1
        index = self.osim_model.istep + 1
        qd = extract_data_frame_by_index(self.q_d, index)
        ud = extract_data_frame_by_index(self.u_d, index)
        ad = extract_data_frame_by_index(self.a_d, index)

        # calculate errors using normal coordinate order (as actuator)
        error_q = []
        error_u = []
        error_a = []  # no error just ad
        for coordinate_name in self.osim_model.get_coordinate_names():
            error_q.append(qd[coordinate_name] - q[coordinate_name])
            error_u.append(ud[coordinate_name] - u[coordinate_name])
            error_a.append(ad[coordinate_name])

        # calculate integral error
        if self.integral_error is None:  # initial condition
            self.integral_error = error_q
        else:  # integration step
            self.integral_error = [
                self.integral_error[i] + error_q[i] * self.osim_model.step_size
                for i in range(len(error_q))
            ]

        if self.use_id:
            t = obs_dict['time']

            # PD: tau_pd = Kp error_q + Kd error_u + Ki integral error_q
            tau_pd = np.dot(self.Kp, error_q) + \
                np.dot(self.Kd, error_u) + \
                np.dot(self.Ki, self.integral_error)

            # get q_s, u_s, a_d in multibody order
            q_s = []  # q simulated
            u_s = []  # u simulated
            a_d = []  # a desired
            for coordinate_name in self.osim_model.get_coordinate_names_multibody_order(
            ):
                q_s.append(q[coordinate_name])
                u_s.append(u[coordinate_name])
                a_d.append(ad[coordinate_name])

            # M qddot + f = tau = M qddot_d + f + M (Kp error_q + Kd error_u)
            # error_a + Kd error_u + Kp error_q = 0
            tau_pd_mo = tau_pd[
                self.normal_to_multibody_order]  # map to multibody order
            M_tau_pd_mo = np.array(
                self.id_model.multiplyByM(t, q_s, tau_pd_mo.tolist()))
            tau_id_mo = np.array(
                self.id_model.calculateResidualForces(t, q_s, u_s, a_d))
            tau = np.add(M_tau_pd_mo[self.multibody_to_normal_order],
                         tau_id_mo[self.multibody_to_normal_order])
            # tau_pd[0] = 0.0     # pelvis_tilt
            tau_pd[1] = 0.0  # pelvis_tx
            tau_pd[2] = 0.0  # pelvis_ty

        elif self.use_stable_pd:
            t = obs_dict['time']

            # PD: tau_pd = Kp error_q + Kd error_u + Ki integral error_q
            tau_pd = np.dot(self.Kp, error_q) + \
                np.dot(self.Kd, error_u) + \
                np.dot(self.Ki, self.integral_error)

            # get q_s, u_s, a_d in multibody order
            q_s = []  # q simulated
            u_s = []  # u simulated
            a_d = []  # a desired
            for coordinate_name in self.osim_model.get_coordinate_names_multibody_order(
            ):
                q_s.append(q[coordinate_name])
                u_s.append(u[coordinate_name])
                a_d.append(0.0)
                # a_d.append(ad[coordinate_name])  # this works as well

            # get inertia from OpenSim (slow operation)
            M_osim = opensim.Matrix()
            self.osim_model.model.getMatterSubsystem().calcM(
                self.osim_model.state, M_osim)
            rows = M_osim.nrow()
            cols = M_osim.ncol()
            M = np.zeros([rows, cols])
            for mi in range(rows):
                for mj in range(cols):
                    M[mi][mj] = M_osim.get(mi, mj)

            # pybullet stable PD
            tau_pd_mo = tau_pd[self.normal_to_multibody_order]
            tau_res_mo = np.array(
                self.id_model.calculateResidualForces(t, q_s, u_s, a_d))
            tau_net_mo = tau_pd_mo - tau_res_mo
            step_size = self.osim_model.step_size
            M_bar = M + np.diagflat([self.Kd * step_size] * len(q_s))
            qddot = np.linalg.solve(M_bar, tau_net_mo)
            tau = tau_pd_mo - np.dot(self.Kd * step_size, qddot)
            tau = tau[self.multibody_to_normal_order]

            # tau[0] = 0.0     # pelvis_tilt
            tau[1] = 0.0  # pelvis_tx
            tau[2] = 0.0  # pelvis_ty
        else:
            # PD: tau_pd = ad + Kp error_q + Kd error_u + Ki integral error_q
            tau_pd = np.dot(1, error_a) + \
                np.dot(self.Kp, error_q) + \
                np.dot(self.Kd, error_u) + \
                np.dot(self.Ki, self.integral_error)
            tau_pd[0] = 0.0  # pelvis_tilt
            tau_pd[1] = 0.0  # pelvis_tx
            tau_pd[2] = 0.0  # pelvis_ty
            tau = tau_pd

        return super(PositionControlEnv, self).step(tau, obs_as_dict)

    def reset(self, obs_as_dict=False):
        self.osim_model.reset()
        self.state_cache = None

        # set initial state
        index = 0
        self.osim_model.set_time(self.q_d.iloc[index]['time'])
        self.osim_model.set_coordinates(
            extract_data_frame_by_index(self.q_d, index))
        self.osim_model.set_velocities(
            extract_data_frame_by_index(self.u_d, index))

        # return observation
        if obs_as_dict:
            return self.get_observation_dict()
        else:
            return self.get_observation()

    def get_state_dict(self):
        obs = {}
        obs.update(self.osim_model.calc_joint_kinematics())
        obs.update(self.osim_model.calc_body_kinematics())
        return obs

    def is_done(self):
        obs = self.get_observation_dict()
        com = obs['body_pos']['center_of_mass']

        if com[1] < 0.75:
            return True
        elif self.osim_model.istep > self.N:
            return True
        else:
            return False

    def get_reward(self):
        return None, None


# %%
# main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--visualize',
                        action='store_true',
                        help='enable simbody visualizer')
    parser.add_argument('-d',
                        '--use_inverse_dynamics',
                        action='store_true',
                        help='use inverse dynamics or PID')
    parser.add_argument('-s',
                        '--use_stable_pd',
                        action='store_true',
                        help='use Stable PD as DeepMimic')

    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='output directory for reading/writing results')

    return parser.parse_args()


def main():
    args = parse_args()
    visualize = args.visualize
    use_id = args.use_inverse_dynamics
    use_stable_pd = args.use_stable_pd
    output = args.output
    max_actuation = 1000

    # step size is set to 0.001 (larger or smaller values do not give good
    # results) because we are in contact with the ground thus must avoid
    # instabilities
    step_size = 0.001

    # this speeds up simulation
    integration_accuracy = 1e-3

    # model and kinematic data (input model same as SCONE to avoid
    # inconsistencies)
    subject_dir = os.path.abspath('../data/2D/')
    model_input_file = os.path.join(subject_dir, 'model/model_generic.osim')
    model_predictive_file = os.path.join(subject_dir,
                                         'scale/model_predictive.osim')
    q_desired_file = os.path.join(subject_dir,
                                  'kinematics_analysis/task_Kinematics_q.sto')
    u_desired_file = os.path.join(subject_dir,
                                  'kinematics_analysis/task_Kinematics_u.sto')

    # remove muscles and add coordinate actuators
    model_file = model_predictive_file
    convert_model_to_torque_actuated(model_input_file, model_predictive_file,
                                     max_actuation, False)

    def make_env():
        return PositionControlEnv(model_file, step_size, integration_accuracy,
                                  visualize, q_desired_file, u_desired_file,
                                  max_actuation, use_id, use_stable_pd)

    env = make_env()
    env.reset()
    while True:
        observation, reward, done, info = env.step(None)
        if done:
            break

    env.osim_model.save_simulation(output)


if __name__ == '__main__':
    main()

# %%
