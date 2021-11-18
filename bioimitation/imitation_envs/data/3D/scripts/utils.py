# Utility functions.
#
# author: Dimitar Stanev <jimstanev@gmail.com>
# %%
import re
import os
import opensim
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText


def calculate_emg_linear_envelope(x, f_sampling=1000, f_band_low=30,
                                  f_band_high=300, f_env=6, to_normalize=True):
    """Calculates the EMG linear envelope by applying the following
    transformations to the raw signal:

    1) Remove mean
    2) Band-pass 4th order Butterworth filter to remove low and high frequencies
    3) Full rectification (use of abs)
    4) Normalization based on max value (if to_normalize=True)
    5) Low-pass filter to calculate the envelope

    """
    f_nyq = f_sampling / 2
    # 1) remove mean
    y = x - x.mean()
    # 2) band-pass
    b, a = signal.butter(4, [f_band_low / f_nyq, f_band_high / f_nyq], 'band')
    y = signal.filtfilt(b, a, y)
    # 3) rectify
    y = np.abs(y)
    # 4) normalize
    if to_normalize:
        y = y / y.max()

    # 5) low-pass
    b, a = signal.butter(2, f_env / f_nyq, 'low')
    env = signal.filtfilt(b, a, y)
    # # plot
    # plt.figure()
    # plt.plot(y, label='raw')
    # plt.plot(env, label='envelop')
    # plt.legend()
    # return
    return env


def normalize_interpolate_dataframe(df, interp_column='time', method='linear'):
    """Normalizes time between [0, 1] and then re-samples data frame at
    constant interval.

    """
    # normalize between 0, 1
    time_old = df.time.to_numpy()
    time_new = (time_old - time_old[0]) / (time_old[-1] - time_old[0])
    df.loc[:, 'time'] = time_new
    # re-sample time with specific interval
    df = df.set_index(interp_column)
    at = np.arange(0, 1.01, 0.01)
    df = df.reindex(df.index | at)
    df = df.interpolate(method=method).loc[at]
    df = df.reset_index()
    df = df.rename(columns={'index': interp_column})
    return df

def osim_vector_to_list(array):
    """Convert SimTK::Vector to Python list.
    """
    temp = []
    for i in range(array.size()):
        temp.append(array[i])

    return temp


def vector_vec3_to_nparray(vector):
    temp = []
    for i in range(vector.size()):
        temp.append([vector[i][0], vector[i][1], vector[i][2]])

    return np.array(temp)


def osim_array_to_list(array):
    """Convert OpenSim::Array<T> to Python list.
    """
    temp = []
    for i in range(array.getSize()):
        temp.append(array.get(i))

    return temp


def list_to_osim_array_str(list_str):
    """Convert Python list of strings to OpenSim::Array<string>."""
    arr = opensim.ArrayStr()
    for element in list_str:
        arr.append(element)

    return arr


def np_array_to_simtk_matrix(array):
    """Convert numpy array to SimTK::Matrix"""
    n, m = array.shape
    M = opensim.Matrix(n, m)
    for i in range(n):
        for j in range(m):
            M.set(i, j, array[i, j])

    return M


def rotate_data_table(table, axis, deg):
    """Rotate OpenSim::TimeSeriesTableVec3 entries using an axis and angle.

    Parameters
    ----------
    table: OpenSim.common.TimeSeriesTableVec3

    axis: 3x1 vector

    deg: angle in degrees

    """
    R = opensim.Rotation(np.deg2rad(deg),
                         opensim.Vec3(axis[0], axis[1], axis[2]))
    for i in range(table.getNumRows()):
        vec = table.getRowAtIndex(i)
        vec_rotated = R.multiply(vec)
        table.setRowAtIndex(i, vec_rotated)


def mm_to_m(table, label):
    """Scale from units in mm for units in m.

    Parameters
    ----------
    label: string containing the name of the column you want to convert

    """
    c = table.updDependentColumn(label)
    for i in range(c.size()):
        c[i] = opensim.Vec3(c[i][0] * 0.001, c[i][1] * 0.001, c[i][2] * 0.001)


def mirror_z(table, label):
    """Mirror the z-component of the vector.

    Parameters
    ----------
    label: string containing the name of the column you want to convert

    """
    c = table.updDependentColumn(label)
    for i in range(c.size()):
        c[i] = opensim.Vec3(c[i][0], c[i][1], -c[i][2])


def lowess_bell_shape_kern(x, y, tau=0.0005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> y_est Locally weighted
    regression: fits a nonparametric regression curve to a scatterplot. The
    arrays x and y contain an equal number of elements; each pair (x[i], y[i])
    defines a data point in the scatterplot. The function returns the estimated
    (smooth) values of y.  The kernel function is the bell shaped function with
    parameter tau. Larger tau will result in a smoother curve.

    """
    n = len(x)
    y_est = np.zeros(n)

    # initializing all weights from the bell shape kernel function
    w = np.array([np.exp(- (x - x[i]) ** 2 / (2 * tau)) for i in range(n)])

    # looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                      [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = np.linalg.solve(A, b)
        y_est[i] = theta[0] + theta[1] * x[i]

    return y_est


def create_opensim_storage(time, data, column_names):
    """Creates a OpenSim::Storage.

    Parameters
    ----------
    time: SimTK::Vector

    data: SimTK::Matrix

    column_names: list of strings

    Returns
    -------
    sto: OpenSim::Storage

    """
    sto = opensim.Storage()
    sto.setColumnLabels(list_to_osim_array_str(['time'] + column_names))
    for i in range(data.nrow()):
        row = opensim.ArrayDouble()
        for j in range(data.ncol()):
            row.append(data.getElt(i, j))

        sto.append(time[i], row)

    return sto


def annotate_plot(ax, text):
    """Annotate a figure by adding a text.
    """
    at = AnchoredText(text, frameon=True, loc='upper left')
    at.patch.set_boxstyle('round, pad=0, rounding_size=0.2')
    ax.add_artist(at)


def rmse_metric(s1, s2):
    """Root mean squared error between two time series.

    """
    # Signals are sampled with the same sampling frequency. Here time
    # series are first aligned.
    # if s1.index[0] < 0:
    #     s1.index = s1.index - s1.index[0]

    # if s2.index[0] < 0:
    #     s2.index = s2.index - s2.index[0]

    t1_0 = s1.index[0]
    t1_f = s1.index[-1]
    t2_0 = s2.index[0]
    t2_f = s2.index[-1]
    t_0 = np.round(np.max([t1_0, t2_0]), 3)
    t_f = np.round(np.min([t1_f, t2_f]), 3)
    x = s1[(s1.index >= t_0) & (s1.index <= t_f)].to_numpy()
    y = s2[(s2.index >= t_0) & (s2.index <= t_f)].to_numpy()
    return np.round(np.sqrt(np.mean((x - y) ** 2)), 3)


def refine_ground_reaction_wrench(data_table, label_triplet, stance_threshold,
                                  tau, debug=True):
    """Clean and filter raw ground reaction forces at a single leg as specified by
    label triplet. This algorithm checks when the foot is in touch with the
    ground (stance phase). When the foot is not in touch then the original data
    contain noise with very small SNR. Therefore, the data is either set to zero
    or to nan. Then, the data is interpolated in case of nan. Finally, the
    signals are low pass filtered using lowess_bell_shape_kern.

    Parameters
    ----------

    data_table: OpenSim::DataTable<Vec3> containing [force, point, moment] for
    each leg

    label_triplet: column identifiers for the wrench triplet (e.g., ['f1', 'p1', 'm1'])

    stance_threshold: values to consider the foot in touch with the ground

    tau: kernel standard divination (filtering)

    debug: Boolean to visualize filtering result

    Returns
    -------

    This function mutates the original data_table

    """
    # get data of single leg
    t = np.array(data_table.getIndependentColumn())
    f = data_table.updDependentColumn(label_triplet[0])
    p = data_table.updDependentColumn(label_triplet[1])
    m = data_table.updDependentColumn(label_triplet[2])
    f_l = vector_vec3_to_nparray(f)
    p_l = vector_vec3_to_nparray(p)
    m_l = vector_vec3_to_nparray(m)

    # debugging
    if debug:
        plt.figure()
        f1 = plt.gca()
        f1.plot(t, f_l)
        plt.figure()
        f2 = plt.gca()
        f2.plot(t, p_l)
        plt.figure()
        f3 = plt.gca()
        f3.plot(t, m_l)

    # remove information when the foot is not touching the ground
    t0 = None
    tf = None
    for i in range(len(f_l)):
        # remove noise
        if f_l[i, 1] < stance_threshold:
            for j in range(3):
                f_l[i, j] = 0
                p_l[i, j] = np.nan
                m_l[i, j] = 0

        # detect heel strike
        if t0 is None and f_l[i, 1] >= stance_threshold:
            t0 = t[i]

        # detect toe off
        if tf is None and t0 is not None and f_l[i, 1] <= stance_threshold:
            tf = t[i]

    # interpolate nan values for points and moments
    f_l = pd.DataFrame(f_l).interpolate(limit_direction="both", kind="cubic").to_numpy()
    p_l = pd.DataFrame(p_l).interpolate(limit_direction="both", kind="cubic").to_numpy()
    m_l = pd.DataFrame(m_l).interpolate(limit_direction="both", kind="cubic").to_numpy()

    # filter data
    for j in range(3):
        # f_l[:, j] = signal.medfilt(f_l[:, j], median)
        f_l[:, j] = lowess_bell_shape_kern(t, f_l[:, j], tau)
        p_l[:, j] = lowess_bell_shape_kern(t, p_l[:, j], tau)
        m_l[:, j] = lowess_bell_shape_kern(t, m_l[:, j], tau)

    # debugging
    if debug:
        f1.plot(t, f_l)
        f2.plot(t, p_l)
        f3.plot(t, m_l)

    # update columns in the original data
    for i in range(f_l.shape[0]):
        f[i] = opensim.Vec3(f_l[i, 0], f_l[i, 1], f_l[i, 2])
        p[i] = opensim.Vec3(p_l[i, 0], p_l[i, 1], p_l[i, 2])
        m[i] = opensim.Vec3(m_l[i, 0], m_l[i, 1], m_l[i, 2])

    return t0, tf, p_l.mean(axis=0)

def read_from_storage(file_name, sampling_interval=0.01,
                      to_filter=False):
    """Read OpenSim.Storage files.

    Parameters
    ----------
    file_name: (string) path to file

    sampling_interval: resample the data with a given interval (0.01)

    to_filter: use low pass 4th order FIR filter with 6Hz cut off
    frequency

    Returns
    -------
    df: pandas data frame

    """
    sto = opensim.Storage(file_name)
    sto.resampleLinear(sampling_interval)
    if to_filter:
        sto.lowpassFIR(4, 6)

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


def index_containing_substring(list_str, pattern):
    """For a given list of strings finds the index of the element that
    contains the substring.

    Parameters
    ----------
    list_str: list of str

    pattern: str
         pattern


    Returns
    -------
    indices: list of int
         the indices where the pattern matches

    """
    return [i for i, item in enumerate(list_str)
            if re.search(pattern, item)]


def plot_sto_file(file_name, plot_file, plots_per_row=4, pattern=None,
                  title_function=lambda x: x):
    """Plots the .sto file (OpenSim) by constructing a grid of subplots.

    Parameters
    ----------
    sto_file: str
        path to file
    plot_file: str
        path to store result
    plots_per_row: int
        subplot columns
    pattern: str, optional, default=None
        plot based on pattern (e.g. only pelvis coordinates)
    title_function: lambda
        callable function f(str) -> str
    """
    df = read_from_storage(file_name)
    labels = df.columns.to_list()
    data = df.to_numpy()

    if pattern is not None:
        indices = index_containing_substring(labels, pattern)
    else:
        indices = range(1, len(labels))

    n = len(indices)
    ncols = int(plots_per_row)
    nrows = int(np.ceil(float(n) / plots_per_row))
    pages = int(np.ceil(float(nrows) / ncols))
    if ncols > n:
        ncols = n

    with PdfPages(plot_file) as pdf:
        for page in range(0, pages):
            fig, ax = plt.subplots(nrows=ncols, ncols=ncols,
                                   figsize=(8, 8))
            ax = ax.flatten()
            for pl, col in enumerate(indices[page * ncols ** 2:page *
                                             ncols ** 2 + ncols ** 2]):
                ax[pl].plot(data[:, 0], data[:, col])
                ax[pl].set_title(title_function(labels[col]))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()


def adjust_model_mass(model_file, mass_change):
    """Given a required mass change adjust all body masses accordingly.

    """
    rra_model = opensim.Model(model_file)
    rra_model.setName('model_adjusted')
    state = rra_model.initSystem()
    current_mass = rra_model.getTotalMass(state)
    new_mass = current_mass + mass_change
    mass_scale_factor = new_mass / current_mass
    for body in rra_model.updBodySet():
        body.setMass(mass_scale_factor * body.getMass())

    # save model with adjusted body masses
    rra_model.printToXML(model_file)


def replace_thelen_muscles_with_millard(model_file, target_folder):
    """Replaces Thelen muscles with Millard muscles so that we can disable
    tendon compliance and perform MuscleAnalysis to compute normalized
    fiber length/velocity without spikes.

    """
    model = opensim.Model(model_file)
    new_force_set = opensim.ForceSet()
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        force = force_set.get(i)
        muscle = opensim.Muscle.safeDownCast(force)
        millard_muscle = opensim.Millard2012EquilibriumMuscle.safeDownCast(
            force)
        thelen_muscle = opensim.Thelen2003Muscle.safeDownCast(force)
        if muscle is None:
            new_force_set.adoptAndAppend(force.clone())
        elif millard_muscle is not None:
            millard_muscle = millard_muscle.clone()
            millard_muscle.set_ignore_tendon_compliance(True)
            new_force_set.adoptAndAppend(millard_muscle)
        elif thelen_muscle is not None:
            millard_muscle = opensim.Millard2012EquilibriumMuscle()
            # properties
            millard_muscle.set_default_activation(
                thelen_muscle.getDefaultActivation())
            millard_muscle.set_activation_time_constant(
                thelen_muscle.get_activation_time_constant())
            millard_muscle.set_deactivation_time_constant(
                thelen_muscle.get_deactivation_time_constant())
            # millard_muscle.set_fiber_damping(0)
            # millard_muscle.set_tendon_strain_at_one_norm_force(
            #     thelen_muscle.get_FmaxTendonStrain())
            millard_muscle.setName(thelen_muscle.getName())
            millard_muscle.set_appliesForce(thelen_muscle.get_appliesForce())
            millard_muscle.setMinControl(thelen_muscle.getMinControl())
            millard_muscle.setMaxControl(thelen_muscle.getMaxControl())
            millard_muscle.setMaxIsometricForce(
                thelen_muscle.getMaxIsometricForce())
            millard_muscle.setOptimalFiberLength(
                thelen_muscle.getOptimalFiberLength())
            millard_muscle.setTendonSlackLength(
                thelen_muscle.getTendonSlackLength())
            millard_muscle.setPennationAngleAtOptimalFiberLength(
                thelen_muscle.getPennationAngleAtOptimalFiberLength())
            millard_muscle.setMaxContractionVelocity(
                thelen_muscle.getMaxContractionVelocity())
            # millard_muscle.set_ignore_tendon_compliance(
            #     thelen_muscle.get_ignore_tendon_compliance())
            millard_muscle.set_ignore_tendon_compliance(True)
            millard_muscle.set_ignore_activation_dynamics(
                thelen_muscle.get_ignore_activation_dynamics())
            # muscle path
            pathPointSet = thelen_muscle.getGeometryPath().getPathPointSet()
            geomPath = millard_muscle.updGeometryPath()
            for j in range(pathPointSet.getSize()):
                pathPoint = pathPointSet.get(j).clone()
                geomPath.updPathPointSet().adoptAndAppend(pathPoint)

            # append
            new_force_set.adoptAndAppend(millard_muscle)
        else:
            raise RuntimeError(
                'cannot handle the type of muscle: ' + force.getName())

    new_force_set.printToXML(os.path.join(target_folder, 'muscle_set.xml'))


def subject_specific_isometric_force(generic_model_file, subject_model_file,
                                     height_generic, height_subject):
    """Adjust the max isometric force of the subject-specific model based on results
    from Handsfield et al. 2014 [1] (equation from Fig. 5A). Function adapted
    from Rajagopal et al. 2015 [2].

    Given the height and mass of the generic and subject models, we can
    calculate the total muscle volume [1]:

    V_total = 47.05 * mass * height + 1289.6

    Since we can calculate the muscle volume and the optimal fiber length of the
    generic and subject model, respectively, we can calculate the force scale
    factor to scale the maximum isometric force of each muscle:

    scale_factor = (V_total_subject / V_total_generic) / (l0_subject / l0_generic)

    F_max_i = scale_factor * F_max_i

    [1] http://dx.doi.org/10.1016/j.jbiomech.2013.12.002
    [2] http://dx.doi.org/10.1109/TBME.2016.2586891

    """
    model_generic = opensim.Model(generic_model_file)
    state_generic = model_generic.initSystem()
    mass_generic = model_generic.getTotalMass(state_generic)

    model_subject = opensim.Model(subject_model_file)
    state_subject = model_subject.initSystem()
    mass_subject = model_subject.getTotalMass(state_subject)

    # formula for total muscle volume
    V_total_generic = 47.05 * mass_generic * height_generic + 1289.6
    V_total_subject = 47.05 * mass_subject * height_subject + 1289.6

    for i in range(0, model_subject.getMuscles().getSize()):
        muscle_generic = model_generic.updMuscles().get(i)
        muscle_subject = model_subject.updMuscles().get(i)

        l0_generic = muscle_generic.getOptimalFiberLength()
        l0_subject = muscle_subject.getOptimalFiberLength()

        force_scale_factor = (V_total_subject / V_total_generic) / (l0_subject /
                                                                    l0_generic)
        muscle_subject.setMaxIsometricForce(force_scale_factor *
                                            muscle_subject.getMaxIsometricForce())

    model_subject.printToXML(subject_model_file)
