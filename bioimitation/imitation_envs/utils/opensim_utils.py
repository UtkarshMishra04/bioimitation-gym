import copy
import opensim
import numpy as np
import pandas as pd
import warnings


def list3_to_vec3(list3):
    """Python list of 3 elements to opensim.Vec3 converter"""
    assert(len(list3) == 3)
    return opensim.Vec3(list3[0], list3[1], list3[2])


def get_default_contact_sphere_parameters():
    """Gets the default contact sphere parameters for typical OpenSim gait
    model.

    Returns
    -------
    parameters: [dictionary]

    """
    return copy.copy(
        {'heel_r': {
            'body': 'calcn_r',
            'location': [0.03, 0.02, 0],
            'orientation': [0, 0, 0],
            'radius': 0.05},
         'toe1_r': {
             'body': 'toes_r',
             'location': [0.02, -0.005, -0.026],
             'orientation': [0, 0, 0],
             'radius': 0.025},
         'toe2_r': {
             'body': 'toes_r',
             'location': [0.02, -0.005, 0.026],
             'orientation': [0, 0, 0],
             'radius': 0.025},
         'heel_l': {
             'body': 'calcn_l',
             'location': [0.03, 0.02, 0],
             'orientation': [0, 0, 0],
             'radius': 0.05},
         'toe1_l': {
             'body': 'toes_l',
             'location': [0.02, -0.005, -0.026],
             'orientation': [0, 0, 0],
             'radius': 0.025},
         'toe2_l': {
             'body': 'toes_l',
             'location': [0.02, -0.005, 0.026],
             'orientation': [0, 0, 0],
             'radius': 0.025}})


def get_default_contact_force_parameters():
    """Gets the default contact force parameters for typical OpenSim gait
    model.

    Returns
    -------
    parameters: [dictionary]

    """
    return copy.copy(
        {'foot_r': {
            'geometries': ['platform', 'heel_r', 'toe1_r', 'toe2_r'],
            'stiffness': 2000000,
            'dissipation': 1.0,
            'static_friction': 0.8,
            'dynamic_friction': 0.8,
            'viscous_friction': 0.6,
            'transition_velocity': 0.1},
         'foot_l': {
             'geometries': ['platform', 'heel_l', 'toe1_l', 'toe2_l'],
             'stiffness': 2000000,
             'dissipation': 1.0,
             'static_friction': 0.8,
             'dynamic_friction': 0.8,
             'viscous_friction': 0.6,
             'transition_velocity': 0.1}})


def get_default_coordinate_limit_force_parameters():
    """Gets the default coordinate limit force parameters for typical
    OpenSim gait model.

    Returns
    -------
    parameters: [dictionary]

    """
    return copy.copy(
        {'hip_flexion_limit_r': {
            'coordinate': 'hip_flexion_r',
            'upper_stiffness': 20,
            'upper_limit': 120,
            'lower_stiffness': 20,
            'lower_limit': -30,
            'damping': 0.25,
            'transition': 10},
         'hip_flexion_limit_l': {
             'coordinate': 'hip_flexion_l',
             'upper_stiffness': 20,
             'upper_limit': 120,
             'lower_stiffness': 20,
             'lower_limit': -30,
             'damping': 0.25,
             'transition': 10},
         'knee_limit_r': {
             'coordinate': 'knee_angle_r',
             'upper_stiffness': 20,
             'upper_limit': 0,
             'lower_stiffness': 20,
             'lower_limit': -140,
             'damping': 0.25,
             'transition': 10},
         'knee_limit_l': {
             'coordinate': 'knee_angle_l',
             'upper_stiffness': 20,
             'upper_limit': 0,
             'lower_stiffness': 20,
             'lower_limit': -140,
             'damping': 0.25,
             'transition': 10},
         'ankle_limit_r': {
             'coordinate': 'ankle_angle_r',
             'upper_stiffness': 20,
             'upper_limit': 20,
             'lower_stiffness': 20,
             'lower_limit': -40,
             'damping': 0.25,
             'transition': 10},
         'ankle_limit_l': {
             'coordinate': 'ankle_angle_l',
             'upper_stiffness': 20,
             'upper_limit': 20,
             'lower_stiffness': 20,
             'lower_limit': -40,
             'damping': 0.25,
             'transition': 10}})


def add_contact_model(model,
                      contact_sphere_parameters,
                      contact_force_parameters):
    """Setup contact geometries and forces for predicting the ground
    reaction forces. The model is adapted internally.

    """
    # handle platform separately
    platform = opensim.ContactHalfSpace()
    platform.setName('platform')
    platform.setBody(model.getGround())
    platform.setLocation(opensim.Vec3(0, 0, 0))
    platform.setOrientation(opensim.Vec3(0, 0, -np.pi / 2))
    model.addContactGeometry(platform)

    # handle contact spheres
    for key, value in contact_sphere_parameters.items():
        sphere = opensim.ContactSphere()
        sphere.setName(key)
        sphere.setBody(model.getBodySet().get(value['body']))
        sphere.setLocation(list3_to_vec3(value['location']))
        sphere.setOrientation(list3_to_vec3(value['orientation']))
        sphere.setRadius(value['radius'])
        model.addContactGeometry(sphere)

    # add contact forces
    for key, value in contact_force_parameters.items():
        force = opensim.HuntCrossleyForce()
        force.setName(key)
        for geometry in value['geometries']:
            force.addGeometry(geometry)

        force.setStiffness(value['stiffness'])
        force.setDissipation(value['dissipation'])
        force.setStaticFriction(value['static_friction'])
        force.setDynamicFriction(value['dynamic_friction'])
        force.setViscousFriction(value['viscous_friction'])
        force.setTransitionVelocity(value['transition_velocity'])
        model.addForce(force)


def add_coordinate_limit_forces(model,
                                coordinate_limit_force_parameters):
    """Setups coordinate limit forces that during forward dynamic
    analysis.

    """
    for key, value in coordinate_limit_force_parameters.items():
        force = opensim.CoordinateLimitForce()
        force.setName(key)
        force.set_coordinate(value['coordinate'])
        force.setUpperStiffness(value['upper_stiffness'])
        force.setUpperLimit(value['upper_limit'])
        force.setLowerStiffness(value['lower_stiffness'])
        force.setLowerLimit(value['lower_limit'])
        force.setDamping(value['damping'])
        force.setTransition(value['transition'])
        model.addForce(force)


def construct_predictive_model(model_file, model_file_out):
    """Adds contact sphere on feet and coordinate limiting forces."""
    # load OpenSim model
    model = opensim.Model(model_file)

    # add contact and coordinate limit forces
    add_contact_model(model,
                      get_default_contact_sphere_parameters(),
                      get_default_contact_force_parameters())
    add_coordinate_limit_forces(model,
                                get_default_coordinate_limit_force_parameters())

    # set pelvis_ty default coordinate so that the model is over the ground plane
    model.updCoordinateSet().get('pelvis_ty').setDefaultValue(1.02)

    # save the new model
    model.finalizeConnections()
    model.setName('model_predictive')
    model.printToXML(model_file_out)


def constraint_model_into_2D(model_file, model_file_out):
    """Lock some of the degrees of freedom to make a planar 2D gait model."""
    model = opensim.Model(model_file)
    locked_coordinates = ['pelvis_list', 'pelvis_rotation', 'pelvis_tz',
                          'hip_adduction_r', 'hip_adduction_l',
                          'hip_rotation_r', 'hip_rotation_l',
                          'lumbar_extension', 'lumbar_bending',
                          'lumbar_rotation']
    for coordinate in locked_coordinates:
        model.updCoordinateSet().get(coordinate).set_locked(True)

    model.printToXML(model_file_out)

def convert_model_to_torque_actuated(model_file, model_file_out,
                                     max_actuation, remove_floating_base=True):
    """Removes muscles and adds coordinate actuators for each degree of freedom."""
    model = opensim.Model(model_file)

    # remove muscles
    muscle_names = opensim.ArrayStr()
    model.getMuscles().getNames(muscle_names)
    for i in range(muscle_names.getSize()):
        model.updForceSet().remove(model.updForceSet().getIndex(muscle_names.get(i)))

    # add coordinate actuators
    for coordinate in model.updCoordinateSet():
        # pelvis tx, ty, tz should not be actuated
        if (remove_floating_base and
            coordinate.getName() in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']) or \
            coordinate.get_locked():
            continue

        # construct coordinate actuators
        act = opensim.CoordinateActuator()
        act.setCoordinate(coordinate)
        act.setName(coordinate.getName() + '_actuator')
        act.setOptimalForce(1)
        act.setMinControl(-max_actuation)
        act.setMaxControl(max_actuation)
        model.addForce(act)
        print("Actuator added for coordinate: " + coordinate.getName())

    # save the new model
    # model.finalizeConnections()
    model.setName('model_predictive_no_muscles')
    model.printToXML(model_file_out)


def osim_array_to_list(array):
    """Convert OpenSim::Array<T> to Python list.
    """
    temp = []
    for i in range(array.getSize()):
        temp.append(array.get(i))

    return temp


def read_from_storage(model_file, file_name, sampling_interval):
    """Read OpenSim.Storage files.

    Parameters
    ----------
    file_name: (string) path to file

    Returns
    -------
    tuple: (labels, time, data)

    """
    sto = opensim.Storage(file_name)
    if sto.isInDegrees():
        model = opensim.Model(model_file)
        model.initSystem()
        model.getSimbodyEngine().convertDegreesToRadians(sto)

    sto.resampleLinear(sampling_interval)

    labels = osim_array_to_list(sto.getColumnLabels())
    time = opensim.ArrayDouble()
    sto.getTimeColumn(time)
    time = osim_array_to_list(time)
    data = []
    for i in range(sto.getSize()):
        temp = osim_array_to_list(sto.getStateVector(i).getData())
        temp.insert(0, time[i])
        data.append(temp)

    df = pd.DataFrame(data, columns=labels)
    df.index = df.time
    return df


def extract_data_frame_by_index(data_frame, index):
    """Convenient interface to extract a dictionary of values indexed by
    names from the columns of the data frame.

    """
    df = data_frame.iloc[index]
    df.pop('time')
    return df


def convert_to_relative(observations):

    for key in observations["body_pos"].keys():

        if 'pelvis_tz' in observations["coordinate_pos"].keys():
            observations["body_pos"][key] = [
                observations["body_pos"][key][0] -
                observations["coordinate_pos"]["pelvis_tx"],
                observations["body_pos"][key][1] -
                observations["coordinate_pos"]["pelvis_ty"],
                observations["body_pos"][key][2] -
                observations["coordinate_pos"]["pelvis_tz"]
            ]
        else:
            observations["body_pos"][key] = [
                observations["body_pos"][key][0] -
                observations["coordinate_pos"]["pelvis_tx"],
                observations["body_pos"][key][1] -
                observations["coordinate_pos"]["pelvis_ty"],
                observations["body_pos"][key][2]
            ]

    return observations


def normalize_forces(observations, mass, gravity, height):

    weight = abs(mass * gravity[1])
    moment = weight * height

    for key in observations["contact_forces"]:
        observations["contact_forces"][key][
            0] = observations["contact_forces"][key][0] / weight
        observations["contact_forces"][key][
            1] = observations["contact_forces"][key][1] / weight
        observations["contact_forces"][key][
            2] = observations["contact_forces"][key][2] / weight
        observations["contact_forces"][key][
            3] = observations["contact_forces"][key][3] / moment
        observations["contact_forces"][key][
            4] = observations["contact_forces"][key][4] / moment
        observations["contact_forces"][key][
            5] = observations["contact_forces"][key][5] / moment

    try:
        for key in observations["coordinate_limit_forces"]:
            observations["coordinate_limit_forces"][
                key] = observations["coordinate_limit_forces"][key] / moment
    except Exception as e:
        pass

    return observations

def kinematics_MSE(observations, desired_kinematics, compare_at_index,
                   observation_key):
    """Calculates the mean squared error between the kinematics (position,
    velocity or accelerations) from the observation and the desired
    kinematics (pandas data frame that contains the whole history) at
    a particular time.

    Parameters
    ----------

    observations: a dictionary of observations from osim-rl
    environment

    desired_kinematics: a pandas data frame containing coordinate
    positions, velocities or accelerations

    compare_at_index: integer corresponding to the entry that we would
    like to compare

    observation_key: is used to select either coordinate_pos,
    coordinate_vel, coordinate_acc

    Returns
    -------

    the mean square error

    """
    assert(observation_key in ['coordinate_pos', 'coordinate_vel', 'coordinate_acc'])
    current = observations[observation_key]
    desired = extract_data_frame_by_index(desired_kinematics, compare_at_index).to_dict()
    error = []
    for name, value in current.items():
        error.append(value - desired[name])

    # return mean squared error
    return np.mean(np.power(error, 2))


def body_kinematics_MSE(observations, desired_body_kinematics,
                        compare_at_index, observation_key, body):
    """Calculates the mean squared error between the body kinematics (linear and
    angular position, velocity and orientation) from the observation and the
    desired body kinematics (pandas data frame that contains the whole history)
    at a particular time.

    Parameters
    ----------

    observations: a dictionary of observations from osim-rl
    environment (always expressed in ground frame)

    desired_body_kinematics: a pandas data frame containing the
    desired body kinematics (always expressed in ground frame)

    compare_at_index: integer corresponding to the entry that we would
    like to compare

    observation_key: is used to select either body_pos, body_vel,
    body_acc, body_pos_rot, body_vel_rot, body_acc_rot

    body: the body of interest (e.g., center_of_mass, torso, calcn_r)

    Returns
    -------

    the mean square error

    """
    assert(observation_key in ['body_pos', 'body_vel', 'body_acc',
                               'body_pos_rot', 'body_vel_rot', 'body_acc_rot'])

    body_current = np.array(observations[observation_key][body])

    frame = extract_data_frame_by_index(desired_body_kinematics, compare_at_index)
    body_spatial = frame.filter(regex=body).to_numpy().reshape(-1)

    if observation_key.find('rot') == -1:
        # compare position
        error = body_current - body_spatial[0:3]
    else:
        if body == 'center_of_mass':  # CoM does not have orientation
            return 0

        # compare orientations
        error = body_current - body_spatial[3:6]

    # return mean squared error
    return np.mean(np.power(error, 2))
