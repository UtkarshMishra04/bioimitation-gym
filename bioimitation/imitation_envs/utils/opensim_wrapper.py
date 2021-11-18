import opensim
import numpy as np
import collections


class OsimModel(object):
    def __init__(self, model_path, step_size, integrator_accuracy, visualize):
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)
        self.kinematics = opensim.Kinematics()
        self.kinematics.setModel(self.model)
        self.model.addAnalysis(self.kinematics)
        self.forces = opensim.ForceReporter()
        self.forces.setModel(self.model)
        self.model.addAnalysis(self.forces)
        self.state = self.model.initSystem()
        self.model.setUseVisualizer(visualize)
        self.brain = opensim.PrescribedController()
        self.istep = 0
        self.step_size = step_size

        # get model sets
        self.analysis_set = self.model.getAnalysisSet()
        self.actuator_set = self.model.getActuators()
        self.force_set = self.model.getForceSet()
        self.body_set = self.model.getBodySet()
        self.coordinate_set = self.model.getCoordinateSet()
        self.marker_set = self.model.getMarkerSet()

        self.coordinate_limit_min = []
        self.coordinate_limit_max = []
        for i in range(self.coordinate_set.getSize()):
            coordinate = self.coordinate_set.get(i)
            if coordinate.getName() not in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
                self.coordinate_limit_max.append(coordinate.getRangeMax())
                self.coordinate_limit_min.append(coordinate.getRangeMin())

        # define controller
        self.action_min = []
        self.action_max = []
        for j in range(self.actuator_set.getSize()):
            func = opensim.Constant(0.0)
            actuator = self.actuator_set.get(j)
            self.brain.addActuator(actuator)
            self.brain.prescribeControlForActuator(j, func)
            scalar_actuator = opensim.ScalarActuator_safeDownCast(actuator)
            if scalar_actuator is None:
                # currently only BodyActuator is not handled (not crucial)
                # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Actuator.html
                raise RuntimeError('un-handled type of scalar actuator')
            else:
                self.action_min.append(scalar_actuator.getMinControl())
                self.action_max.append(scalar_actuator.getMaxControl())

        self.action_space_size = self.actuator_set.getSize()
        self.model.addController(self.brain)

        self.coordinate_names = [
            self.coordinate_set.get(i).getName()
            for i in range(self.coordinate_set.getSize())
        ]

        self.is_muscle_model = self.model.getMuscles().getSize() != 0

        if self.is_muscle_model:
            self.muscle_set = self.model.getMuscles()
            self.muscle_names = [
                self.muscle_set.get(i).getName()
                for i in range(self.muscle_set.getSize())
            ]

        self.state = self.model.initSystem()

        # Obtain multibody order, because after v3.3 the state has different
        # order. Unfortunately, self.model.getCoordinatesInMultibodyTreeOrder()
        # not exposed. This solution might not work always TODO.
        temp_dict = {}
        cnt = 0
        for coord in self.coordinate_set:
            mbix = coord.getBodyIndex()
            mqix = coord.getMobilizerQIndex()
            offset = cnt
            if not mqix == 0:
                offset = mqix
                cnt += 1

            temp_dict[mbix + offset] = coord.getName()

        temp_dict_ord = collections.OrderedDict(sorted(temp_dict.items()))
        self.coordinate_names_multibody_order = list(temp_dict_ord.values())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            # raise ValueError('NaN passed in the activation vector. ')
            action = np.zeros(action.shape)

        # might have torque actuators
        action = np.clip(np.array(action), self.action_min, self.action_max)
        self.last_action = action

        brain = opensim.PrescribedController.safeDownCast(
            self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(float(action[j]))

    def get_last_action(self):
        return self.last_action

    def get_coordinate_names(self):
        return self.coordinate_names

    def get_coordinate_names_multibody_order(self):
        return self.coordinate_names_multibody_order

    def calc_joint_kinematics(self):
        self.model.realizeAcceleration(self.state)
        obs = {}
        obs['time'] = self.state.getTime()

        # coordinates
        obs['coordinate_pos'] = {}
        obs['coordinate_vel'] = {}
        obs['coordinate_acc'] = {}
        for i in range(self.coordinate_set.getSize()):
            coordinate = self.coordinate_set.get(i)
            name = coordinate.getName()
            obs['coordinate_pos'][name] = coordinate.getValue(self.state)
            obs['coordinate_vel'][name] = coordinate.getSpeedValue(self.state)
            obs['coordinate_acc'][name] = coordinate.getAccelerationValue(
                self.state)

        return obs

    def calc_body_kinematics(self):
        self.model.realizeAcceleration(self.state)
        obs = {}
        obs['time'] = self.state.getTime()

        # bodies
        obs['body_pos'] = {}
        obs['body_vel'] = {}
        obs['body_acc'] = {}
        obs['body_pos_rot'] = {}
        obs['body_vel_rot'] = {}
        obs['body_acc_rot'] = {}
        for i in range(self.body_set.getSize()):
            body = self.body_set.get(i)
            name = body.getName()
            obs['body_pos'][name] = [
                body.getTransformInGround(self.state).p()[i] for i in range(3)
            ]
            obs['body_vel'][name] = [
                body.getVelocityInGround(self.state).get(1).get(i)
                for i in range(3)
            ]
            obs['body_acc'][name] = [
                body.getAccelerationInGround(self.state).get(1).get(i)
                for i in range(3)
            ]

            obs['body_pos_rot'][name] = [
                body.getTransformInGround(
                    self.state).R().convertRotationToBodyFixedXYZ().get(i)
                for i in range(3)
            ]
            obs['body_vel_rot'][name] = [
                body.getVelocityInGround(self.state).get(0).get(i)
                for i in range(3)
            ]
            obs['body_acc_rot'][name] = [
                body.getAccelerationInGround(self.state).get(0).get(i)
                for i in range(3)
            ]

        # mass center
        obs['body_pos']['center_of_mass'] = [
            self.model.calcMassCenterPosition(self.state)[i] for i in range(3)
        ]
        obs['body_vel']['center_of_mass'] = [
            self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)
        ]
        obs['body_acc']['center_of_mass'] = [
            self.model.calcMassCenterAcceleration(self.state)[i]
            for i in range(3)
        ]

        return obs

    def calc_forces_info(self):
        self.model.realizeDynamics(self.state)
        obs = {}
        obs['time'] = self.state.getTime()

        # forces
        obs['forces'] = {}
        obs['contact_forces'] = {}
        obs['coordinate_limit_forces'] = {}
        obs['scalar_actuator_forces'] = {}
        for i in range(self.force_set.getSize()):
            force = self.force_set.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            # we check the type of force for quick access
            contact_force = opensim.HuntCrossleyForce_safeDownCast(force)
            coordinate_limit_force = opensim.CoordinateLimitForce_safeDownCast(
                force)
            scalar_actuator = opensim.ScalarActuator_safeDownCast(force)
            if contact_force:
                # It is assumed that the first 6 values is the total
                # wrench (force, moment) applied on the ground plane
                # (they must be negated). The rest are the forces
                # applied on individual points.
                obs['contact_forces'][name] = [
                    -values.get(0), -values.get(1), -values.get(2),
                    -values.get(3), -values.get(4), -values.get(5)
                ]
            elif coordinate_limit_force:
                # coordinate limiting forces return two values, but
                # only one is active (non-zero) or both are zero
                if values.get(0) == 0:
                    value = values.get(1)
                else:
                    value = values.get(0)

                obs['coordinate_limit_forces'][name] = value
            elif scalar_actuator:
                obs['scalar_actuator_forces'][name] = values.get(0)
            else:
                obs['forces'][name] = [
                    values.get(i) for i in range(values.size())
                ]

        return obs

    def calc_muscles_info(self):
        # self.model.realizeVelocity(self.state)  # no need to realize state
        obs = {}
        obs['time'] = self.state.getTime()

        # muscles (model might be torque actuated)
        if self.model.getMuscles().getSize() != 0:
            obs['muscles'] = {}
            for i in range(self.model.getMuscles().getSize()):
                muscle = self.model.getMuscles().get(i)
                name = muscle.getName()
                obs['muscles'][name] = {}
                obs['muscles'][name]['activation'] = muscle.getActivation(
                    self.state)
                obs['muscles'][name]['fiber_length'] = muscle.getFiberLength(
                    self.state)
                obs['muscles'][name][
                    'fiber_velocity'] = muscle.getFiberVelocity(self.state)
                obs['muscles'][name]['fiber_force'] = muscle.getFiberForce(
                    self.state)

        return obs

    def calc_markers_info(self):
        self.model.realizeAcceleration(self.state)
        obs = {}
        obs['time'] = self.state.getTime()

        # markers
        obs['markers'] = {}
        for i in range(self.marker_set.getSize()):
            marker = self.marker_set.get(i)
            name = marker.getName()
            obs['markers'][name] = {}
            obs['markers'][name]['pos'] = [
                marker.getLocationInGround(self.state)[i] for i in range(3)
            ]
            obs['markers'][name]['vel'] = [
                marker.getVelocityInGround(self.state)[i] for i in range(3)
            ]
            obs['markers'][name]['acc'] = [
                marker.getAccelerationInGround(self.state)[i] for i in range(3)
            ]

        return obs

    def get_action_space_size(self):
        return self.action_space_size

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.model.equilibrateMuscles(self.state)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.model.initializeState()
        self.state.setTime(0)
        self.istep = 0
        self.reset_manager()

    def integrate(self):
        self.istep += 1
        self.state = self.manager.integrate(self.step_size * self.istep)

    def set_time(self, t):
        self.state.setTime(t)
        # for istep we assume that time starts from 0
        self.istep = int(self.state.getTime() / self.step_size)
        self.reset_manager()

    def set_coordinates(self, q_dict):
        '''Set coordinate values.
        Parameters
        ----------
        q_dict: a dictionary containing the coordinate names and
        values in rad or m.
        '''
        for coordinate, value in q_dict.items():
            self.coordinate_set.get(coordinate).setValue(self.state, value)

        self.reset_manager()

    def set_velocities(self, u_dict):
        '''Set coordinate velocities.
        Parameters
        ----------
        u_dict: a dictionary containing the coordinate names and
        velocities in rad/s or m/s.
        '''
        for coordinate, value in u_dict.items():
            self.coordinate_set.get(coordinate).setSpeedValue(
                self.state, value)

        self.reset_manager()

    def save_simulation(self, base_dir):
        '''Saves simulation files into base_dir.'''
        self.manager.getStateStorage().printToFile(
            base_dir + '/simulation_States.sto', 'w', '')
        self.analysis_set.printResults('simulation', base_dir)
