# Utility functions.
#
# author: Dimitar Stanev <jimstanev@gmail.com>
import re
import os
import opensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
import matplotlib.gridspec as grid_spec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import holoviews as hv


def perform_muscle_analysis(model_file, state_file, output_dir):
    """Perform OpenSim MuscleAnalysis on SCONE state file generated
    through simulation. This might be used to calculate joint moments
    induced by muscles.

    """
    model = opensim.Model(model_file)

    # construct static optimization
    state_storage = opensim.Storage(state_file)
    muscle_analysis = opensim.MuscleAnalysis()
    muscle_analysis.setStartTime(state_storage.getFirstTime())
    muscle_analysis.setEndTime(state_storage.getLastTime())
    model.addAnalysis(muscle_analysis)

    # analysis
    analysis = opensim.AnalyzeTool(model)
    analysis.setName('muscle_analysis')
    analysis.setModel(model)
    analysis.setInitialTime(state_storage.getFirstTime())
    analysis.setFinalTime(state_storage.getLastTime())
    analysis.setStatesFileName(state_file)
    # analysis.setLowpassCutoffFrequency(6)
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(os.path.abspath(output_dir))
    analysis.run()


def annotate_plot(ax, text, loc='upper left'):
    """Annotate a figure by adding a text.
    """
    at = AnchoredText(text, frameon=True, loc=loc)
    at.patch.set_boxstyle('round, pad=0, rounding_size=0.2')
    at.patch.set_alpha(0.2)
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
    return np.round(np.sqrt(np.mean((x - y) ** 2)), 2)


def simtk_vec_to_list(vec):
    """Convert SimTK::Vec_<T> to Python list.
    """
    temp = []
    for i in range(vec.size()):
        temp.append(vec[i])

    return temp


def osim_array_to_list(array):
    """Convert OpenSim::Array<T> to Python list.
    """
    temp = []
    for i in range(array.getSize()):
        temp.append(array.get(i))

    return temp


def to_gait_cycle(data_frame, t0, tf):
    temp = data_frame[(data_frame.time >= t0) & (data_frame.time <= tf)].copy()
    temp.time = data_frame['time'].transform(lambda x: 100.0 / (tf - t0) * (x - t0))
    temp.set_index('time', inplace=True)
    temp.index.names = ['gait cycle (%)']
    return temp


def read_from_storage(file_name, to_filter=False):
    """Read OpenSim.Storage files.

    Parameters
    ----------
    file_name: (string) path to file

    Returns
    -------
    tuple: (labels, time, data)
    """
    sto = opensim.Storage(file_name)
    sto.resampleLinear(0.01)
    if to_filter:
        sto.lowpassFIR(4, 6)

    labels = osim_array_to_list(sto.getColumnLabels())
    time = opensim.ArrayDouble()
    sto.getTimeColumn(time)
    time = np.round(osim_array_to_list(time), 3)
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
                # make very small number zero before plotting
                data[np.abs(data[:, col]) < 1e-9, col] = 0
                ax[pl].plot(data[:, 0], data[:, col])
                ax[pl].set_title(title_function(labels[col]))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()

def add_emg_on_off_ong(ax, var_name):
    """Adds EMG on/off regions.

    Parameters
    ----------

    ax : Matplotlib <AxesSubplot>
        Axes subplot object to plot on
    var_name : string
        Name of the variable to plot

    Returns
    -------
    None

    """
    color = 'k'
    if "glut_max" in var_name:
        ax.axhline(y=1.0, xmin=0.00, xmax=0.25, color=color, linewidth=10)
        ax.axhline(y=1.0, xmin=0.95, xmax=1.00, color=color, linewidth=10)
    if "psoas" in var_name:
        ax.axhline(y=1.0, xmin=0.65, xmax=0.75, color=color, linewidth=10)
    if "hamstrings" in var_name:
        ax.axhline(y=1.0, xmin=0.00, xmax=0.20, color=color, linewidth=10)
        ax.axhline(y=1.0, xmin=0.80, xmax=1.00, color=color, linewidth=10)
    if "bifemsh" in var_name:
        ax.axhline(y=1.0, xmin=0.65, xmax=0.85, color=color, linewidth=10)
    if "vast" in var_name:
        ax.axhline(y=1.0, xmin=0.00, xmax=0.20, color=color, linewidth=10)
        ax.axhline(y=1.0, xmin=0.85, xmax=1.00, color=color, linewidth=10)
    if "rect_fem" in var_name:
        ax.axhline(y=1.0, xmin=0.55, xmax=0.65, color=color, linewidth=10)
    if "tib_ant" in var_name:
        ax.axhline(y=1.0, xmin=0.00, xmax=0.15, color=color, linewidth=10)
        ax.axhline(y=1.0, xmin=0.60, xmax=1.00, color=color, linewidth=10)
    if "gas" in var_name:
        ax.axhline(y=1.0, xmin=0.05, xmax=0.50, color=color, linewidth=10)
    if "sol" in var_name:
        ax.axhline(y=1.0, xmin=0.05, xmax=0.55, color=color, linewidth=10)


def add_healthy_range_schwartz(ax, var_name):
    """This method plots the value ranges of 'var_name' from healthy
    clinical observations.

    Source (free gait): https://doi.org/10.1016/j.jbiomech.2008.03.015

    Parameters
    ----------

    ax : Matplotlib <AxesSubplot>
        Axes subplot object to plot on
    var_name : string
        Name of the variable to plot

    Returns
    -------
    None

    """
    norm_min = None
    norm_mean = None
    norm_max = None
    if "pelvis_tilt" in var_name:
        norm_min = np.multiply([7.2, 7.1, 7.0, 6.7, 6.4, 6.1, 6.0,
                                5.9, 5.9, 6.0, 6.2, 6.4, 6.6, 6.8,
                                6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5,
                                7.5, 7.5, 7.4, 7.2, 7.1, 6.9, 6.7,
                                6.5, 6.2, 5.9, 5.7, 5.7, 5.7, 5.8,
                                5.9, 6.1, 6.3, 6.5, 6.6, 6.7, 6.8,
                                7.0, 7.1, 7.2, 7.3, 7.4, 7.3, 7.2,
                                7.1, 6.9 ], -1)
        norm_mean = np.multiply([12.2, 12.0, 11.8, 11.6, 11.2, 10.9,
                                 10.8, 10.7, 10.7, 10.8, 11.0, 11.2,
                                 11.4, 11.5, 11.6, 11.7, 11.8, 12.0,
                                 12.1, 12.3, 12.4, 12.5, 12.5, 12.4,
                                 12.2, 12.0, 11.9, 11.7, 11.4, 11.1,
                                 10.8, 10.6, 10.5, 10.6, 10.7, 10.9,
                                 11.0, 11.2, 11.4, 11.5, 11.6, 11.7,
                                 11.8, 12.0, 12.2, 12.3, 12.4, 12.4,
                                 12.3, 12.1, 11.9 ], -1)
        norm_max = np.multiply([17.2, 17.0, 16.7, 16.4, 16.0, 15.7,
                                15.5, 15.5, 15.5, 15.7, 15.9, 16.0,
                                16.2, 16.3, 16.4, 16.5, 16.6, 16.7,
                                16.9, 17.2, 17.3, 17.4, 17.4, 17.3,
                                17.2, 17.0, 16.8, 16.6, 16.3, 16.0,
                                15.7, 15.5, 15.4, 15.4, 15.6, 15.8,
                                16.0, 16.1, 16.2, 16.3, 16.4, 16.5,
                                16.7, 16.9, 17.1, 17.3, 17.4, 17.4,
                                17.3, 17.2, 17.0 ], -1)
    if "pelvis_rotation" in var_name:
        norm_min = np.multiply([0.4, 0.5, 0.3, 0.0, -0.2, -0.4, -0.3, 
                                -0.2, -0.2, -0.2, -0.4, -0.6, -1.0, 
                                -1.4, -1.8, -2.3, -2.8, -3.4, -4.0, 
                                -4.7, -5.5, -6.2, -6.9, -7.5, -7.9, 
                                -8.2, -8.2, -8.0, -7.7, -7.4, -7.3, 
                                -7.3, -7.4, -7.5, -7.4, -7.1, -6.7, 
                                -6.2, -5.7, -5.1, -4.6, -4.0, -3.5, 
                                -3.0, -2.4, -1.8, -1.2, -0.7, -0.2, 
                                0.2, 0.4], 1)
        norm_mean = np.multiply([4.4, 4.4, 4.3, 3.9, 3.6, 3.4, 3.4, 
                                3.5, 3.6, 3.5, 3.3, 3.0, 2.6, 2.1, 
                                1.6, 1.1, 0.5, 0.0, -0.6, -1.2, -1.9, 
                                -2.6, -3.2, -3.7, -4.1, -4.4, -4.4, 
                                -4.2, -3.9, -3.6, -3.5, -3.5, -3.6, 
                                -3.7, -3.6, -3.4, -3.1, -2.7, -2.2, 
                                -1.7, -1.2, -0.6, -0.1, 0.5, 1.1, 1.8, 
                                2.4, 3.1, 3.6, 4.0, 4.2], 1)
        norm_max = np.multiply([8.4, 8.4, 8.2, 7.8, 7.4, 7.2, 7.2, 7.3, 
                                7.3, 7.2, 6.9, 6.5, 6.1, 5.5, 5.0, 4.5, 
                                3.9, 3.4, 2.9, 2.3, 1.7, 1.1, 0.5, 0.0, 
                                -0.3, -0.6, -0.6, -0.5, -0.2, 0.1, 0.3, 
                                0.3, 0.2, 0.2, 0.2, 0.3, 0.6, 0.9, 1.3, 
                                1.8, 2.3, 2.8, 3.3, 3.9, 4.6, 5.4, 6.1, 
                                6.8, 7.4, 7.8, 8.0], 1)
    if "pelvis_list" in var_name:
        norm_min = np.multiply([-1.2, -0.9, -0.5, 0.1, 0.7, 1.2, 1.7, 1.9, 
                                2.0, 1.8, 1.3, 0.7, 0.1, -0.6, -1.2, -1.7, 
                                -2.1, -2.3, -2.5, -2.5, -2.5, -2.4, -2.3, 
                                -2.3, -2.4, -2.6, -3.0, -3.5, -4.1, -4.8, 
                                -5.5, -6.0, -6.3, -6.3, -6.1, -5.6, -5.0, 
                                -4.3, -3.7, -3.0, -2.5, -2.1, -1.8, -1.7, 
                                -1.6, -1.7, -1.7, -1.8, -1.7, -1.6, -1.4], 1)
        norm_mean = np.multiply([0.8, 1.1, 1.5, 2.1, 2.8, 3.4, 3.8, 
                                4.1, 4.1, 3.8, 3.4, 2.8, 2.1, 1.5, 
                                0.8, 0.3, -0.1, -0.3, -0.5, -0.5, -0.5, 
                                -0.4, -0.3, -0.4, -0.5, -0.7, -1.0, -1.4, 
                                -2.0, -2.7, -3.3, -3.8, -4.1, -4.1, -3.9, 
                                -3.4, -2.8, -2.2, -1.5, -0.9, -0.4, 0.0, 
                                0.2, 0.4, 0.4, 0.4, 0.3, 0.2, 0.3, 0.4, 
                                0.6], 1)
        norm_max = np.multiply([2.8, 3.1, 3.5,  4.2, 4.9, 5.5, 6.0, 6.2, 
                                6.2, 5.9, 5.4, 4.8, 4.2, 3.5, 2.9, 2.4, 2.0, 
                                1.7, 1.5, 1.5, 1.5, 1.6, 1.6, 1.6, 1.5, 1.3, 1.0, 
                                0.6, 0.0, -0.6, -1.1, -1.6, -1.8, -1.9, -1.7, -1.3, 
                                -0.7, 0.0, 0.6, 1.2, 1.7, 2.0, 2.3, 2.4, 2.4, 2.4, 
                                2.3, 2.2, 2.3, 2.4, 2.6], 1)
    if "hip_flexion" in var_name:
        norm_min = [30.4, 30.2, 29.9, 29.3, 28.3, 26.9, 25.1, 23.1,
                    20.9, 18.7, 16.5, 14.3, 12.1, 9.9, 7.6, 5.4, 3.2,
                    1.0, -1.1, -3.2, -5.1, -6.9, -8.5, -10.0, -11.2,
                    -12.1, -12.6, -12.5, -11.7, -10.1, -7.7, -4.4,
                    -0.5, 3.6, 7.8, 11.7, 15.4, 18.7, 21.7, 24.3,
                    26.5, 28.3, 29.7, 30.6, 31.1, 31.2, 31.0, 30.6,
                    30.3, 30.1, 29.9 ]
        norm_mean = [35.8, 35.7, 35.5, 35.1, 34.2, 32.8, 31.1, 29.1,
                     26.9, 24.7, 22.5, 20.3, 18.1, 15.8, 13.6, 11.4,
                     9.3, 7.3, 5.3, 3.4, 1.6, -0.1, -1.7, -3.2, -4.4,
                     -5.3, -5.8, -5.7, -4.9, -3.4, -0.9, 2.3, 6.1,
                     10.1, 14.1, 17.9, 21.4, 24.6, 27.4, 29.9, 32.0,
                     33.7, 35.1, 36.0, 36.6, 36.7, 36.5, 36.2, 35.8,
                     35.6, 35.5 ]
        norm_max = [41.3, 41.2, 41.1, 40.9, 40.2, 38.8, 37.1, 35.0,
                    32.9, 30.7, 28.5, 26.3, 24.0, 21.8, 19.6, 17.4,
                    15.4, 13.6, 11.8, 10.0, 8.3, 6.7, 5.1, 3.7, 2.5,
                    1.5, 1.0, 1.1, 1.8, 3.4, 5.8, 9.0, 12.7, 16.6,
                    20.4, 24.0, 27.4, 30.5, 33.2, 35.5, 37.5, 39.1,
                    40.5, 41.4, 42.0, 42.2, 42.1, 41.7, 41.3, 41.0,
                    41.0 ]
    if "hip_adduction" in var_name:
        norm_min = np.multiply([-5.1, -4.4, -3.5, -2.3, -1.0, 0.2, 1.0, 1.5, 1.8,
                    1.8, 1.7, 1.4, 1.1, 0.7, 0.4, 0.1, -0.2, -0.3, -0.4,
                    -0.5, -0.5, -0.6, -0.8, -1.1, -1.6, -2.3, -3.1, -4.2, 
                    -5.6, -7.1, -8.5, -9.5, -10.1, -10.1, -9.7, -9.1, -8.4,
                    -7.7, -7.0, -6.4, -5.9, -5.5, -5.2, -5.1, -5.1, -5.2,
                     -5.4, -5.6, -5.7, -5.5, -5.0], -1)
        norm_mean = np.multiply([-1.6, -0.9, 0.1, 1.3, 2.6, 3.7, 4.5, 5.0, 5.2, 
                    5.2,5.0,4.7,4.4,4.0,3.6,3.2,2.9,2.7,2.6,2.5,2.4,
                    2.4,2.2,1.9,1.4,0.8,0.0,-1.1,-2.3,-3.8,-5.2,-6.3,
                    -7.0,-7.1,-6.8,-6.3,-5.6,-4.9,-4.2,-3.6,-3.1,-2.7,
                    -2.4,-2.2,-2.2,-2.2,-2.3,-2.4,-2.4,-2.1,-1.6], -1)
        norm_max = np.multiply([2.0, 2.6, 3.6, 4.8, 6.1, 7.2, 7.9, 8.4, 8.6, 8.6, 
                    8.4, 8.0, 7.6, 7.2, 6.7, 6.3, 6.0, 5.7, 5.6, 5.5, 
                    5.4, 5.4, 5.2, 4.9, 4.4, 3.8, 3.1, 2.1, 0.9, -0.5, 
                    -1.9, -3.1, -3.8, -4.1, -3.9, -3.5, -2.8, -2.1, -1.4, 
                    -0.8, -0.3, 0.1, 0.4, 0.6, 0.7, 0.8, 0.8, 0.8, 0.9, 
                    1.3, 1.8], -1)
    if "knee_angle" in var_name:
        norm_min = [-0.1, 2.5, 5.3, 8.3, 10.8, 12.3, 13.0, 12.9, 12.4,
                    11.6, 10.6, 9.4, 8.2, 7.0, 5.8, 4.6, 3.5, 2.5,
                    1.6, 0.9, 0.5, 0.5, 1.1, 2.2, 3.8, 6.0, 8.7, 12.2,
                    16.4, 21.5, 27.3, 33.6, 39.9, 45.7, 50.0, 52.5,
                    53.3, 52.4, 50.1, 46.5, 42.0, 36.5, 30.3, 23.7,
                    16.8, 10.2, 4.4, 0.2, -2.0, -1.9, -0.2 ]
        norm_mean = [5.6, 7.9, 10.9, 14.1, 16.9, 18.6, 19.3, 19.2,
                     18.6, 17.6, 16.5, 15.2, 13.9, 12.6, 11.3, 10.1,
                     9.0, 8.0, 7.2, 6.6, 6.2, 6.3, 6.8, 7.9, 9.5,
                     11.6, 14.5, 18.1, 22.6, 27.9, 33.9, 40.2, 46.3,
                     51.7, 55.7, 58.2, 59.1, 58.7, 56.9, 53.9, 49.8,
                     44.7, 38.8, 32.4, 25.5, 18.7, 12.5, 7.6, 4.6,
                     4.0, 5.4 ]
        norm_max = [11.2, 13.4, 16.5, 20.0, 23.0, 24.8, 25.6, 25.5,
                    24.8, 23.7, 22.4, 21.0, 19.6, 18.2, 16.8, 15.6,
                    14.6, 13.6, 12.9, 12.3, 12.0, 12.0, 12.5, 13.5,
                    15.1, 17.3, 20.2, 24.0, 28.8, 34.3, 40.5, 46.8,
                    52.8, 57.7, 61.4, 63.8, 65.0, 65.0, 63.8, 61.3,
                    57.6, 52.9, 47.3, 41.0, 34.2, 27.2, 20.5, 14.9,
                    11.2, 10.0, 11.0 ]
    if "ankle_angle" in var_name:
        norm_min = [-7.7, -9.1, -10.3, -9.9, -8.2, -6.1, -4.2, -2.5,
                    -1.0, 0.2, 1.4, 2.3, 3.1, 3.7, 4.3, 4.8, 5.3, 5.7, 6.1, 6.4,
                    6.7, 7.0, 7.1, 6.9, 6.3, 4.8, 2.1, -2.2, -8.6, -16.2, -23.3,
                    -27.6, -28.6, -26.7, -22.9, -18.7, -14.7, -11.1, -8.1, -5.6,
                    -3.7, -2.4, -1.7, -1.7, -2.4, -3.4, -4.6, -5.7, -6.4, -7.0,
                    -7.8]
        norm_mean = [-2.1, -3.9, -5.6, -5.6, -4.1, -2.1, -0.2, 1.5,
                     2.9, 4.1, 5.2, 6.1, 6.8, 7.4, 8.0, 8.6, 9.1, 9.7,
                     10.2, 10.7, 11.1, 11.5, 11.8, 11.8, 11.5, 10.4,
                     8.4, 4.9, -0.2, -6.9, -13.8, -18.5, -19.8, -18.0,
                     -14.7, -11.2, -7.9, -4.9, -2.4, -0.3, 1.1, 2.1,
                     2.5, 2.4, 1.8, 1.0, 0.1, -0.6, -1.1, -1.4, -2.3]
        norm_max = [3.5, 1.3, -0.8, -1.3, -0.1, 1.9, 3.8, 5.5, 6.9,
                    8.0, 9.0, 9.8, 10.5, 11.1, 11.8, 12.4, 13.0, 13.7,
                    14.3, 15.0, 15.5, 16.1, 16.5, 16.8, 16.7, 16.0,
                    14.6, 12.1, 8.1, 2.4, -4.3, -9.5, -10.9, -9.3,
                    -6.5, -3.7, -1.1, 1.3, 3.4, 4.9, 6.0, 6.6, 6.7,
                    6.4, 5.9, 5.4, 4.8, 4.5, 4.3, 4.1, 3.1]
    if "grf_norm_x" in var_name:
        norm_min = [-0.01, -0.08, -0.15, -0.17, -0.20, -0.21, -0.19,
                    -0.16, -0.13, -0.11, -0.09, -0.07, -0.06, -0.05, -0.04, -0.03,
                    -0.02, -0.01, 0.00, 0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.14,
                    0.15, 0.14, 0.08, 0.01, -0.01, -0.01, -0.01, 0.00, 0.00, 0.00,
                    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                    0.00, 0.00, 0.00, 0.00, 0.00]
        norm_mean = [0.00, -0.03, -0.09, -0.12, -0.15, -0.16, -0.15,
                     -0.13, -0.11, -0.08, -0.07, -0.05, -0.04, -0.03, -0.02, -0.02,
                     -0.01, 0.00, 0.02, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18,
                     0.19, 0.18, 0.14, 0.07, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00,
                     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                     0.00, 0.00, 0.00, 0.00, 0.00]
        norm_max = [0.01, 0.02, -0.03, -0.07, -0.10, -0.12, -0.11,
                    -0.10, -0.08, -0.06, -0.04, -0.03, -0.02, -0.01, -0.01, 0.00,
                    0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.21,
                    0.23, 0.22, 0.19, 0.12, 0.05, 0.01, 0.01, 0.00, 0.00, 0.00,
                    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                    0.00, 0.00, 0.00, 0.00, 0.00]
    if "grf_norm_y" in var_name:
        norm_min = [0.05, 0.61, 0.76, 0.86, 1.00, 1.15, 1.22, 1.22,
                    1.17, 1.10, 1.03, 0.96, 0.92, 0.89, 0.89, 0.89,
                    0.91, 0.93, 0.96, 1.01, 1.06, 1.11, 1.16, 1.19,
                    1.19, 1.15, 1.06, 0.90, 0.65, 0.36, 0.15, 0.05,
                    0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                    0.00, 0.00, 0.00]
        norm_mean = [0.01, 0.47, 0.60, 0.69, 0.84, 0.99, 1.08, 1.10,
                     1.06, 1.00, 0.93, 0.87, 0.82, 0.80, 0.79, 0.80,
                     0.81, 0.84, 0.87, 0.91, 0.96, 1.01, 1.06, 1.08,
                     1.07, 1.02, 0.91, 0.71, 0.45, 0.21, 0.07, 0.02,
                     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                     0.00, 0.00, 0.00]
        norm_max= [-0.03, 0.33, 0.43, 0.52, 0.68, 0.83, 0.93, 0.97,
                   0.95, 0.89, 0.82, 0.77, 0.73, 0.71, 0.70, 0.70,
                   0.72, 0.74, 0.78, 0.82, 0.86, 0.91, 0.95, 0.97,
                   0.96, 0.89, 0.75, 0.53, 0.25, 0.05, -0.01, -0.01,
                   -0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00]
    if 'hamstrings' in var_name:
        norm_min = np.clip([0.12, 0.11, 0.09, 0.07, 0.06, 0.04, 0.03,
                            0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                            0.00, 0.00, -0.01, -0.01, -0.02, -0.02,
                            -0.02, -0.02, -0.02, -0.03, -0.03, -0.03,
                            -0.03, -0.04, -0.04, -0.04, -0.03, -0.03,
                            -0.03, -0.04, -0.04, -0.04, -0.03, -0.03,
                            -0.02, -0.02, -0.01, 0.01, 0.04, 0.06,
                            0.09, 0.11, 0.12, 0.13, 0.13, 0.12, 0.11],
                           a_min=0, a_max=1)
        norm_mean = [0.30, 0.28, 0.26, 0.24, 0.21, 0.18, 0.16, 0.14,
                     0.14, 0.14, 0.15, 0.15, 0.15, 0.15, 0.14, 0.14,
                     0.14, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.11,
                     0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.05,
                     0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07,
                     0.09, 0.12, 0.16, 0.20, 0.24, 0.28, 0.31, 0.32,
                     0.32, 0.30, 0.28]
        norm_max = [0.47, 0.45, 0.43, 0.41, 0.36, 0.32, 0.28, 0.27,
                    0.27, 0.28, 0.28, 0.29, 0.29, 0.29, 0.29, 0.28,
                    0.28, 0.28, 0.28, 0.28, 0.27, 0.27, 0.25, 0.24,
                    0.22, 0.20, 0.18, 0.18, 0.17, 0.16, 0.15, 0.14,
                    0.14, 0.15, 0.15, 0.16, 0.16, 0.15, 0.15, 0.16,
                    0.19, 0.23, 0.28, 0.33, 0.39, 0.45, 0.49, 0.51,
                    0.50, 0.48, 0.46]
    if 'rect_fem' in var_name:
        norm_min = np.clip([0.08, 0.10, 0.12, 0.12, 0.11, 0.09, 0.06,
                            0.04, 0.02, 0.01, 0.00, 0.00, 0.00, -0.01,
                            -0.01, -0.02, -0.02, -0.02, -0.02, -0.02,
                            -0.01, -0.01, -0.02, -0.02, -0.02, -0.02,
                            -0.01, 0.01, 0.01, 0.00, -0.01, -0.02,
                            -0.02, -0.03, -0.03, -0.03, -0.03, -0.04,
                            -0.04, -0.04, -0.04, -0.03, -0.03, -0.02,
                            -0.01, 0.01, 0.02, 0.03, 0.05, 0.06,
                            0.08], a_min=0, a_max=1)
        norm_mean = [0.22, 0.27, 0.31, 0.32, 0.30, 0.25, 0.20, 0.16,
                     0.13, 0.11, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07,
                     0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08,
                     0.09, 0.09, 0.10, 0.10, 0.11, 0.12, 0.13, 0.14,
                     0.13, 0.12, 0.11, 0.09, 0.08, 0.08, 0.07, 0.07,
                     0.06, 0.06, 0.06, 0.07, 0.09, 0.10, 0.12, 0.14,
                     0.17, 0.20, 0.24]
        norm_max = [0.36, 0.43, 0.50, 0.52, 0.49, 0.42, 0.34, 0.28,
                    0.24, 0.21, 0.18, 0.17, 0.16, 0.16, 0.16, 0.16,
                    0.16, 0.16, 0.16, 0.15, 0.15, 0.16, 0.17, 0.19,
                    0.20, 0.21, 0.20, 0.20, 0.21, 0.24, 0.27, 0.29,
                    0.29, 0.27, 0.24, 0.22, 0.20, 0.19, 0.18, 0.17,
                    0.16, 0.16, 0.16, 0.17, 0.18, 0.20, 0.23, 0.25,
                    0.29, 0.33, 0.39]
    if 'tib_ant' in var_name:
        norm_min = [0.31, 0.32, 0.31, 0.25, 0.18, 0.11, 0.05, 0.02,
                    0.01, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.01,
                    0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.04, 0.03,
                    0.03, 0.01, 0.00, 0.00, 0.01, 0.04, 0.09, 0.15,
                    0.19, 0.23, 0.24, 0.25, 0.24, 0.22, 0.18, 0.15,
                    0.12, 0.09, 0.08, 0.09, 0.10, 0.12, 0.15, 0.19,
                    0.24, 0.28, 0.30]
        norm_mean = [0.48, 0.51, 0.50, 0.44, 0.34, 0.25, 0.19, 0.17,
                     0.15, 0.14, 0.14, 0.13, 0.13, 0.12, 0.11, 0.11,
                     0.11, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13, 0.12,
                     0.11, 0.10, 0.09, 0.10, 0.13, 0.18, 0.25, 0.31,
                     0.36, 0.39, 0.41, 0.42, 0.40, 0.37, 0.32, 0.28,
                     0.23, 0.20, 0.18, 0.19, 0.21, 0.24, 0.28, 0.33,
                     0.39, 0.45, 0.48]
        norm_max = [0.66, 0.70, 0.70, 0.62, 0.50, 0.40, 0.34, 0.31,
                    0.30, 0.29, 0.28, 0.26, 0.25, 0.23, 0.22, 0.21,
                    0.21, 0.20, 0.20, 0.21, 0.21, 0.21, 0.21, 0.21,
                    0.20, 0.18, 0.18, 0.20, 0.25, 0.32, 0.41, 0.48,
                    0.53, 0.56, 0.58, 0.58, 0.57, 0.52, 0.47, 0.40,
                    0.34, 0.30, 0.28, 0.29, 0.32, 0.36, 0.41, 0.47,
                    0.55, 0.61, 0.66]
    if 'gastroc' in var_name:
        norm_min = np.clip([-0.02, -0.01, 0.00, 0.00, 0.00, 0.00,
                            0.00, 0.02, 0.04, 0.07, 0.11, 0.14, 0.17,
                            0.19, 0.21, 0.24, 0.27, 0.30, 0.32, 0.34,
                            0.33, 0.30, 0.26, 0.19, 0.13, 0.07, 0.03,
                            0.01, 0.00, -0.01, -0.01, -0.01, -0.01,
                            -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
                            -0.02, -0.02, -0.02, -0.02, -0.02, -0.03,
                            -0.03, -0.03, -0.03, -0.02, -0.02, -0.01
                            ], a_min=0, a_max=1)
        norm_mean = [0.11, 0.11, 0.11, 0.11, 0.12, 0.13, 0.15, 0.18,
                     0.21, 0.25, 0.28, 0.31, 0.34, 0.36, 0.39, 0.41,
                     0.44, 0.47, 0.50, 0.52, 0.52, 0.50, 0.45, 0.39,
                     0.31, 0.23, 0.16, 0.11, 0.07, 0.06, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09,
                     0.10, 0.10, 0.10]
        norm_max = [0.23, 0.22, 0.22, 0.22, 0.24, 0.27, 0.31, 0.35,
                    0.39, 0.42, 0.45, 0.48, 0.50, 0.53, 0.56, 0.59,
                    0.62, 0.65, 0.68, 0.71, 0.72, 0.70, 0.65, 0.58,
                    0.49, 0.38, 0.29, 0.21, 0.15, 0.12, 0.11, 0.11,
                    0.12, 0.12, 0.12, 0.13, 0.13, 0.12, 0.12, 0.12,
                    0.12, 0.12, 0.13, 0.14, 0.16, 0.17, 0.20, 0.21,
                    0.23, 0.23, 0.22]
    if 'hip_flexion_moment' in var_name:
        norm_min = [-0.01, 0.37, 0.23, 0.33, 0.30, 0.28, 0.26, 0.21,
                    0.15, 0.08, 0.01, -0.02, -0.04, -0.07, -0.09,
                    -0.13, -0.17, -0.23, -0.27, -0.34, -0.42, -0.51,
                    -0.62, -0.72, -0.80, -0.85, -0.84, -0.76, -0.61,
                    -0.50, -0.46, -0.40, -0.33, -0.28, -0.22, -0.17,
                    -0.13, -0.10, -0.08, -0.07, -0.07, -0.08, -0.07,
                    -0.05, 0.00, 0.08, 0.15, 0.20, 0.20, 0.13, -0.04]
        norm_mean = [0.12, 0.73, 0.54, 0.52, 0.50, 0.48, 0.44, 0.39,
                     0.32, 0.24, 0.17, 0.12, 0.09, 0.06, 0.03, 0.00,
                     -0.03, -0.08, -0.12, -0.18, -0.25, -0.33, -0.42,
                     -0.50, -0.58, -0.62, -0.62, -0.55, -0.43, -0.34,
                     -0.32, -0.27, -0.21, -0.17, -0.14, -0.10, -0.08,
                     -0.05, -0.03, -0.03, -0.02, -0.02, 0.00, 0.04,
                     0.11, 0.20, 0.28, 0.33, 0.34, 0.28, 0.12]
        norm_max = [0.26, 1.08, 0.84, 0.72, 0.70, 0.67, 0.63, 0.57,
                    0.48, 0.40, 0.32, 0.27, 0.23, 0.19, 0.16, 0.13,
                    0.10, 0.08, 0.04, -0.02, -0.08, -0.14, -0.21,
                    -0.29, -0.36, -0.40, -0.40, -0.35, -0.25, -0.19,
                    -0.18, -0.14, -0.10, -0.07, -0.05, -0.03, -0.02,
                    0.00, 0.01, 0.02, 0.03, 0.04, 0.08, 0.14, 0.22,
                    0.32, 0.41, 0.47, 0.49, 0.43, 0.28]
    if 'hip_adduction_moment' in var_name:
        norm_min = [-0.06, -0.26,-0.21, -0.07, 0.09, 0.20, 0.27, 
                    0.31, 0.34, 0.35, 0.35, 0.33, 0.30, 0.28, 0.27, 
                    0.26, 0.26, 0.27, 0.28, 0.29, 0.29, 0.29, 0.28, 
                    0.27, 0.24, 0.19, 0.12, 0.01, -0.11, -0.17, -0.18, 
                    -0.15, -0.13, -0.10, -0.08, -0.06, -0.04, -0.03, 
                    -0.03, -0.02, -0.02, -0.03, -0.03, -0.04, -0.05, 
                    -0.06, -0.08, -0.10, -0.11, -0.10, -0.06]
        norm_mean = [0.00, -0.09, -0.05, 0.08, 0.25, 0.36, 0.42, 
                    0.46, 0.48, 0.49, 0.48, 0.45, 0.42, 0.39, 0.37, 
                    0.36, 0.36, 0.37, 0.38, 0.40, 0.42, 0.43, 0.44, 
                    0.44, 0.42, 0.38, 0.30, 0.19, 0.05, -0.06, -0.09, 
                    -0.08, -0.06, -0.05, -0.03, -0.01, 0.00, 0.01, 
                    0.02, 0.02, 0.02, 0.02, 0.01, 0.00, -0.01, -0.01, 
                    -0.02, -0.03, -0.04, -0.04, 0.01]
        norm_max = [0.06, 0.07, 0.11, 0.23, 0.40, 0.52, 0.58, 0.61, 
                    0.63, 0.63, 0.61, 0.57, 0.53, 0.50, 0.47, 0.46, 
                    0.46, 0.46, 0.48, 0.51, 0.54, 0.57, 0.59, 0.60, 
                    0.60, 0.56, 0.48, 0.36, 0.20, 0.05, -0.01, -0.01, 
                    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.06,
                    0.06, 0.06, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 
                    0.03, 0.03, 0.07]
    if 'knee_flexion_moment' in var_name:
        norm_min = [-0.21, -0.47, -0.25, -0.07, 0.03, 0.10, 0.13,
                    0.13, 0.09, 0.04, -0.02, -0.07, -0.13, -0.17,
                    -0.20, -0.24, -0.27, -0.30, -0.34, -0.36, -0.38,
                    -0.39, -0.37, -0.32, -0.25, -0.17, -0.08, 0.00,
                    0.04, 0.04, 0.04, 0.02, 0.01, 0.02, 0.02, 0.01,
                    0.00, -0.01, -0.01, -0.02, -0.03, -0.04, -0.07,
                    -0.10, -0.15, -0.21, -0.28, -0.34, -0.36, -0.32,
                    -0.22]
        norm_mean = [-0.13, -0.29, -0.07, 0.08, 0.23, 0.33, 0.37,
                     0.35, 0.30, 0.24, 0.17, 0.10, 0.03, -0.02, -0.06,
                     -0.10, -0.14, -0.17, -0.20, -0.23, -0.24, -0.24,
                     -0.22, -0.17, -0.10, -0.02, 0.06, 0.12, 0.14,
                     0.12, 0.10, 0.08, 0.08, 0.09, 0.08, 0.07, 0.05,
                     0.03, 0.02, 0.01, 0.00, -0.02, -0.03, -0.06,
                     -0.10, -0.15, -0.20, -0.24, -0.26, -0.22, -0.13]
        norm_max = [-0.05, -0.12, 0.11, 0.22, 0.42, 0.55, 0.60, 0.58,
                    0.52, 0.43, 0.35, 0.27, 0.19, 0.13, 0.08, 0.03,
                    -0.01, -0.04, -0.07, -0.09, -0.10, -0.10, -0.07,
                    -0.02, 0.05, 0.12, 0.20, 0.24, 0.23, 0.19, 0.17,
                    0.15, 0.14, 0.15, 0.14, 0.12, 0.10, 0.07, 0.05,
                    0.03, 0.02, 0.01, 0.00, -0.02, -0.04, -0.08,
                    -0.12, -0.15, -0.16, -0.13, -0.04]
    if 'knee_adduction_moment' in var_name:
        norm_min = [-0.03, -0.12, -0.10, -0.05, 0.04, 0.11, 0.14,
                    0.15, 0.16, 0.15, 0.14, 0.12, 0.10, 0.08, 0.07,
                    0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08,
                    0.09, 0.10, 0.10, 0.09, 0.04, -0.02, -0.06, -0.07,
                    -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.02,
                    -0.01, -0.01, -0.01, -0.01, -0.02, -0.03, -0.03,
                    -0.04, -0.05, -0.06, -0.06, -0.05, -0.03]
        norm_mean = [0.00, -0.05, -0.04, 0.04, 0.14, 0.23, 0.27, 0.28,
                     0.28, 0.27, 0.25, 0.22, 0.19, 0.17, 0.16, 0.14,
                     0.14, 0.14, 0.14, 0.15, 0.16, 0.18, 0.19, 0.21,
                     0.22, 0.21, 0.19, 0.14, 0.06, 0.00, -0.03, -0.03,
                     -0.03, -0.02, -0.01, 0.00, 0.00, 0.00, 0.00,
                     0.01, 0.01, 0.01, 0.00, 0.00, -0.01, -0.01,
                     -0.01, -0.02, -0.01, -0.01, 0.00]
        norm_max = [0.02, 0.03, 0.03, 0.12, 0.25, 0.34, 0.39, 0.41,
                    0.41, 0.39, 0.37, 0.33, 0.29, 0.26, 0.24, 0.23,
                    0.22, 0.22, 0.23, 0.25, 0.27, 0.29, 0.31, 0.32,
                    0.33, 0.32, 0.29, 0.24, 0.15, 0.06, 0.01, 0.00,
                    0.00, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                    0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03,
                    0.03, 0.03, 0.03]
    if 'ankle_angle_moment' in var_name:
        norm_min = [-0.01, -0.14, -0.23, -0.22, -0.19, -0.14, -0.10,
                    -0.06, -0.02, 0.02, 0.08, 0.14, 0.20, 0.27, 0.34, 0.41, 0.48,
                    0.55, 0.63, 0.71, 0.79, 0.87, 0.93, 0.98, 0.99, 0.95, 0.83,
                    0.61, 0.31, 0.05, -0.06, -0.07, -0.06, -0.04, -0.03, -0.02,
                    -0.01, -0.01, -0.01, -0.01, -0.01, -0.02, -0.02, -0.02, -0.02,
                    -0.02, -0.02, -0.01, -0.01, 0.00, -0.01]
        norm_mean = [0.00, -0.08, -0.13, -0.11, -0.06, 0.00, 0.06,
                     0.12, 0.18, 0.25, 0.31, 0.38, 0.44, 0.50, 0.56,
                     0.62, 0.69, 0.76, 0.84, 0.92, 1.00, 1.08, 1.16,
                     1.21, 1.23, 1.19, 1.07, 0.86, 0.56, 0.25, 0.05,
                     -0.02, -0.03, -0.03, -0.02, -0.01, -0.01, -0.01,
                     -0.01, -0.01, -0.01, -0.01, -0.01, -0.02, -0.02,
                     -0.01, -0.01, 0.00, 0.00, 0.00, 0.00]
        norm_max = [0.01, -0.02, -0.03, 0.00, 0.07, 0.15, 0.23, 0.31,
                    0.39, 0.47, 0.55, 0.61, 0.67, 0.72, 0.78, 0.84,
                    0.90, 0.97, 1.04, 1.12, 1.21, 1.30, 1.39, 1.45,
                    1.47, 1.42, 1.31, 1.11, 0.81, 0.46, 0.17, 0.03,
                    -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01,
                    -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01,
                    0.00, 0.00, 0.01, 0.01, 0.01]
    if norm_min is not None and norm_max is not None:
        time_vector = np.linspace(0, 100, len(norm_min))
        ax.fill_between(time_vector, norm_min, norm_max,
                        facecolor='silver', alpha=0.5, label='clin.')

def get_heel_strike_events(grf, side):
    """Get the time instances of heel strike."""
    assert(side in ['r', 'l'])
    if side == 'r':
        leg_name = 'foot_r.ground.force.Y'
    else:
        leg_name = 'foot_l.ground.force.Y'

    # apply moving average to smooth results
    leg = -grf[leg_name].rolling(5).mean()
    max_grf = leg.max()
    leg.dropna(inplace=True)

    # detect heel strike
    threshold = max_grf * 0.25  # used to reject bumps
    swing = False
    heel_strike = []
    for i in range(leg.shape[0]):
        if leg.iloc[i] > threshold:
            if swing:
                heel_strike.append(grf['time'].iloc[i])
                swing = False
        else:
            swing = True

    return heel_strike


def plot_scone_kinematics(state, grf, side, output_file=None):
    """Plot 2D kinematic angles."""
    assert(side in ['r', 'l'])
    # calculate heel strike events
    hs_time = get_heel_strike_events(grf, side)

    # get kinematic variables
    kinematics = state[['time',
                        '/jointset/ground_pelvis/pelvis_tilt/value',
                        '/jointset/ground_pelvis/pelvis_rotation/value',
                        '/jointset/ground_pelvis/pelvis_list/value',
                        '/jointset/' + 'hip_' + side +'/hip_flexion_' + side + '/value',
                        '/jointset/' + 'hip_' + side +'/hip_rotation_' + side + '/value',
                        '/jointset/' + 'hip_' + side +'/hip_adduction_' + side + '/value',
                        '/jointset/' + 'knee_' + side +'/knee_angle_' + side + '/value',
                        '/jointset/' + 'ankle_' + side +'/ankle_angle_' + side + '/value']]

    # plot kinematics
    fig, ax = plt.subplots(3, 3, figsize=(12, 6))
    ax = ax.flatten()

    for i in range(len(hs_time) - 1):
        normalized = to_gait_cycle(kinematics, hs_time[i], hs_time[i+1])
        normalized = normalized.apply(lambda x: np.rad2deg(x))
        normalized['/jointset/ground_pelvis/pelvis_tilt/value'].plot(ax=ax[0])
        normalized['/jointset/ground_pelvis/pelvis_rotation/value'].plot(ax=ax[1])
        normalized['/jointset/ground_pelvis/pelvis_list/value'].plot(ax=ax[2])
        normalized['/jointset/' + 'hip_' + side +'/hip_flexion_' + side + '/value'].plot(ax=ax[3])
        normalized['/jointset/' + 'hip_' + side +'/hip_rotation_' + side + '/value'].plot(ax=ax[4])
        normalized['/jointset/' + 'hip_' + side +'/hip_adduction_' + side + '/value'].plot(ax=ax[5])
        normalized['/jointset/' + 'knee_' + side +'/knee_angle_' + side + '/value'].apply(lambda x: -x).plot(ax=ax[6])
        normalized['/jointset/' + 'ankle_' + side +'/ankle_angle_' + side + '/value'].plot(ax=ax[7])

    ax[0].set_ylabel('pelvis_tilt (deg)')
    ax[0].set_xlabel('gait cycle (%)')
    ax[0].set_xlim([0, 100])
    ax[0].set_xticks(range(0, 120, 20))
    ax[1].set_ylabel('pelvis_rotation (deg)')
    ax[1].set_xlabel('gait cycle (%)')
    ax[1].set_xlim([0, 100])
    ax[1].set_xticks(range(0, 120, 20))
    ax[2].set_ylabel('pelvis_list (deg)')
    ax[2].set_xlabel('gait cycle (%)')
    ax[2].set_xlim([0, 100])
    ax[2].set_xticks(range(0, 120, 20))
    ax[3].set_ylabel('hip_flexion (deg)')
    ax[3].set_xlabel('gait cycle (%)')
    ax[3].set_xlim([0, 100])
    ax[3].set_xticks(range(0, 120, 20))
    ax[4].set_ylabel('hip_rotation (deg)')
    ax[4].set_xlabel('gait cycle (%)')
    ax[4].set_xlim([0, 100])
    ax[4].set_xticks(range(0, 120, 20))
    ax[5].set_ylabel('hip_adduction (deg)')
    ax[5].set_xlabel('gait cycle (%)')
    ax[5].set_xlim([0, 100])
    ax[5].set_xticks(range(0, 120, 20))
    ax[6].set_ylabel('knee_angle (deg)')
    ax[6].set_xlabel('gait cycle (%)')
    ax[6].set_xlim([0, 100])
    ax[6].set_xticks(range(0, 120, 20))
    ax[7].set_ylabel('ankle_angle (deg)')
    ax[7].set_xlabel('gait cycle (%)')
    ax[7].set_xlim([0, 100])
    ax[7].set_xticks(range(0, 120, 20))
    add_healthy_range_schwartz(ax[0], 'pelvis_tilt')
    add_healthy_range_schwartz(ax[1], 'pelvis_rotation')
    add_healthy_range_schwartz(ax[2], 'pelvis_list')
    add_healthy_range_schwartz(ax[3], 'hip_flexion')
    add_healthy_range_schwartz(ax[4], 'hip_rotation')
    add_healthy_range_schwartz(ax[5], 'hip_adduction')
    add_healthy_range_schwartz(ax[6], 'knee_angle')
    add_healthy_range_schwartz(ax[7], 'ankle_angle')
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.kinematics.pdf', bbox_inches='tight')


def plot_scone_vertical_reactions(state, grf, side, model_file, output_file=None):
    """Plot 2D vertical reactions."""
    assert(side in ['r', 'l'])

    # calculate heel strike events
    hs_time = get_heel_strike_events(grf, side)

    if side == 'r':
        leg_name = 'foot_r.ground.force'
    else:
        leg_name = 'foot_l.ground.force'

    # get kinematic variables
    forces = grf[['time', leg_name + '.X', leg_name + '.Y']]

    # calculate body mass
    model = opensim.Model(model_file)
    s = model.initSystem()
    mass = model.getTotalMass(s)

    # plot kinematics
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for i in range(len(hs_time) - 1):
        normalized = to_gait_cycle(forces, hs_time[i], hs_time[i+1])
        normalized[leg_name + '.X'].apply(lambda x: -x / (mass * 9.81)).plot(ax=ax[0])
        normalized[leg_name + '.Y'].apply(lambda x: -x / (mass * 9.81)).plot(ax=ax[1])

    ax[0].set_ylabel('fore-aft')
    ax[0].set_xlabel('gait cycle (%)')
    ax[0].set_xlim([0, 100])
    ax[0].set_xticks(range(0, 120, 20))
    ax[1].set_ylabel('vertical')
    ax[1].set_xlabel('gait cycle (%)')
    ax[1].set_xlim([0, 100])
    ax[1].set_xticks(range(0, 120, 20))

    add_healthy_range_schwartz(ax[0], 'grf_norm_x')
    add_healthy_range_schwartz(ax[1], 'grf_norm_y')
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.grf.pdf', bbox_inches='tight')


def plot_scone_joint_moments(state, grf, model_file, muscle_analysis_output_dir,
                             side, output_file=None):
    """Calculates the joint moments obtained from OpenSim's MuscleAnalysis
    tool. You can use perform_muscle_analysis and to generate the
    necessary files and then call this function passing the correct
    folder.

    """
    assert(side in ['r', 'l'])

    # get all files generated by the analysis tool
    _, _, filenames = next(os.walk(muscle_analysis_output_dir))

    # find moment files of interest
    ankle_moment_file = ''
    knee_moment_file = ''
    hip_flexion_moment_file = ''
    # hip_rotation_moment_file = ''
    hip_adduction_moment_file = ''
    for f in filenames:
        if '_Moment_' + 'ankle_angle_' + side in f:
            ankle_moment_file = f
        if '_Moment_' + 'knee_angle_' + side in f:
            knee_moment_file = f
        if '_Moment_' + 'hip_flexion_' + side in f:
            hip_flexion_moment_file = f
        # if '_Moment_' + 'hip_rotation_' + side in f:
        #     hip_rotation_moment_file = f
        if '_Moment_' + 'hip_adduction_' + side in f:
            hip_adduction_moment_file = f

    # load moment storage
    ankle_moment = read_from_storage(os.path.join(muscle_analysis_output_dir,
                                                  ankle_moment_file))
    knee_moment = read_from_storage(os.path.join(muscle_analysis_output_dir,
                                                 knee_moment_file))
    hip_flexion_moment = read_from_storage(os.path.join(muscle_analysis_output_dir,
                                                hip_flexion_moment_file))
    # hip_rotation_moment = read_from_storage(os.path.join(muscle_analysis_output_dir,
    #                                             hip_rotation_moment_file))
    hip_adduction_moment = read_from_storage(os.path.join(muscle_analysis_output_dir,
                                                hip_adduction_moment_file))

    # get total mass (for normalization)
    model = opensim.Model(model_file)
    s = model.initSystem()
    mass = model.getTotalMass(s)

    # calculate normalized joint moments and place into a DataFrame
    ankle_moment_norm = ankle_moment.sum(axis=1) / mass
    knee_moment_norm = knee_moment.sum(axis=1) / mass
    hip_flexion_moment_norm = hip_flexion_moment.sum(axis=1) / mass
    # hip_rotation_moment_norm = hip_rotation_moment.sum(axis=1) / mass
    hip_adduction_moment_norm = hip_adduction_moment.sum(axis=1) / mass
    moments = pd.concat([hip_flexion_moment_norm, hip_adduction_moment_norm, 
                        knee_moment_norm, ankle_moment_norm], axis=1)
    moments['time'] = moments.index
    moments.columns = ['hip_flexion_moment_' + side, 'hip_adduction_moment_' + side,
                        'knee_angle_moment_' + side, 'ankle_angle_moment_' + side, 'time']

    # calculate heel strike events
    hs_time = get_heel_strike_events(grf, side)

    # plot results
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    ax = ax.flatten()

    for i in range(len(hs_time) - 1):
        normalized = to_gait_cycle(moments, hs_time[i], hs_time[i+1])
        # here we negate hip and ankle to comply with Schwartz ranges
        normalized['hip_flexion_moment_' + side].apply(lambda x: -x).plot(ax=ax[0])
        # normalized['hip_rotation_moment_' + side].apply(lambda x: -x).plot(ax=ax[1])
        normalized['hip_adduction_moment_' + side].apply(lambda x: -x).plot(ax=ax[2])
        normalized['knee_angle_moment_' + side].plot(ax=ax[3])
        normalized['ankle_angle_moment_' + side].apply(lambda x: -x).plot(ax=ax[4])

    ax[0].set_title('hip_flexion')
    ax[0].set_ylabel('<- flexion extension -> \n (N m / kg)')
    ax[0].set_xlabel('gait cycle (%)')
    ax[0].set_xlim([0, 100])
    ax[0].set_xticks(range(0, 120, 20))
    # ax[1].set_title('hip_rotation')
    # ax[1].set_ylabel('<- internal external -> \n (N m / kg)')
    # ax[1].set_xlabel('gait cycle (%)')
    # ax[1].set_xlim([0, 100])
    # ax[1].set_xticks(range(0, 120, 20))
    ax[2].set_title('hip_adduction')
    ax[2].set_ylabel('<- abduction adduction -> \n (N m / kg)')
    ax[2].set_xlabel('gait cycle (%)')
    ax[2].set_xlim([0, 100])
    ax[2].set_xticks(range(0, 120, 20))
    ax[3].set_title('knee')
    ax[3].set_ylabel('<- flexion extension -> \n (N m / kg)')
    ax[3].set_xlabel('gait cycle (%)')
    ax[3].set_xlim([0, 100])
    ax[3].set_xticks(range(0, 120, 20))
    ax[4].set_title('ankle')
    ax[4].set_ylabel('<- dorsiflexion plantarflexion -> \n (N m / kg)')
    ax[4].set_xlabel('gait cycle (%)')
    ax[4].set_xlim([0, 100])
    ax[4].set_xticks(range(0, 120, 20))
    add_healthy_range_schwartz(ax[0], 'hip_flexion_moment')
    # add_healthy_range_schwartz(ax[1], 'hip_rotation_moment')
    add_healthy_range_schwartz(ax[2], 'hip_adduction_moment')
    add_healthy_range_schwartz(ax[3], 'knee_flexion_moment')
    add_healthy_range_schwartz(ax[4], 'ankle_angle_moment')

    fig.delaxes(ax[5])
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.moment.pdf', bbox_inches='tight')


def plot_scone_muscle_excitations(state, grf, muscles, side, col=4, output_file=None):
    """Plot muscle excitations."""
    # get muscle excitations
    mask = ['time']
    for muscle in muscles:
        mask.append('/forceset/' + muscle + '_' + side + '/activation')

    excitations = state[mask]

    # calculate heel strike events
    hs_time = get_heel_strike_events(grf, side)

    if side == 'r':
        leg_name = 'foot_r.ground.force'
    else:
        leg_name = 'foot_l.ground.force'

    # visualize muscle excitations
    M = len(muscles)
    if M % col == 0:
        N = int(np.floor(M / col))
    else:
        N = int(np.floor(M / col)) + 1

    fig, ax = plt.subplots(N, col, figsize=(3 * col, 3 * N))
    ax = ax.flatten()

    # add gait cycle plots
    for i in range(len(hs_time) - 1):
        normalized = to_gait_cycle(excitations, hs_time[i],
                                   hs_time[i+1])
        for j in range(len(muscles)):
            normalized[mask[j + 1]].plot(ax=ax[j])

    # configure subfigures and add range values
    for i in range(len(muscles)):
        add_healthy_range_schwartz(ax[i], mask[i + 1])
        add_emg_on_off_ong(ax[i], mask[i + 1])
        ax[i].set_xlabel('gait cycle (%)')
        muscle_name = mask[i + 1]
        ax[i].set_ylabel(muscle_name[0:muscle_name.find('_' + side + '.')])
        ax[i].set_ylim([0, 1])
        ax[i].set_xlim([0, 100])
        ax[i].set_xticks(range(0, 120, 20))

    # remove unused plots
    i = len(ax) - 1
    if M % col != 0:
        while i >= M:
            fig.delaxes(ax[i])
            i = i - 1

    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.activations.pdf', bbox_inches='tight')


def plot_pattern_formation_weights(parameters, muscles, output_file=None):
    """Plot pattern formation to motor neuron weights."""
    # get pattern formation weights
    pf_w = []
    for muscle in muscles:
        mask = parameters.index.str.contains('^P.*{muscle}.*'.format(muscle=muscle))
        pf_w.append(parameters.loc[mask]['best'].tolist())

    # visualize pattern formation weights
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = ax.imshow(pf_w, cmap='jet')  # vmin=0, vmax=1.0,
    ax.set_xticks(range(len(pf_w[0])))
    ax.set_xticklabels(['P' + str(i) for i in range(1, 6)])
    ax.set_yticks(range(len(pf_w)))
    ax.set_yticklabels(muscles)
    fig.colorbar(pos, ax=ax)
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.patterns.pdf', bbox_inches='tight')


def extract_neuron_activity(state, muscle, segment, neuron, cycle):
    # get columns
    regex = 'time|{muscle}.{segment}.MN.output$|.*->{muscle}.{segment}.{neuron}'\
        .format(muscle=muscle, segment=segment, neuron=neuron)
    mask = state.columns.str.contains(regex)
    df = state.iloc[:, mask]

    # isolate one gait cycle
    hs_time = get_heel_strike_events(state, muscle[-1])
    df = to_gait_cycle(df, hs_time[cycle], hs_time[cycle + 1])

    return df


def plot_neuron_activity(state, muscle, segment, neuron, cycle,
                         output_file=None):
    # isolate specific columns
    df = extract_neuron_activity(state, muscle, segment, neuron, cycle)

    # stack plot
    fig, ax = plt.subplots(figsize=(8, 4))
    # df['sum'] = df.drop(columns=['{muscle}.{segment}.MN.output'\
        #     .format(muscle=muscle, segment=segment, neuron=neuron)]).sum(axis=1)
    area = df.plot.area(stacked=False, ax=ax)
    area.set_ylabel('activity')
    ax.legend(bbox_to_anchor=(1.05, 1.05)) # loc='lower left'
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.' + muscle + '.' + segment + '.' + neuron +
                    '.pdf', bbox_inches='tight')


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a
    distinct RGB color; the keyword argument name must be a standard
    mpl colormap name.

    '''
    return plt.cm.get_cmap(name, n)


def plot_neuron_activity_grid(state, muscle, segment, neuron, cycle,
                              output_file=None):
    # isolate specific columns
    df = extract_neuron_activity(state, muscle, segment, neuron, cycle)

    # rename columns
    df.columns = [name.replace(muscle + '.', '').replace(segment + '.', '')
                  for name in df.columns]

    # columns and color map
    column_names = df.columns
    columns = df.shape[1]
    # cmap = get_cmap(columns, 'RdBu')
    # cmap = lambda i: 'Grey'

    def cmap(series):
        """If signal is positive then red else blue."""
        if np.all(np.sign(series) >= 0):
            return 'r'
        else:
            return 'b'

    # grid specs
    gs = (grid_spec.GridSpec(columns, 1))
    fig = plt.figure(figsize=(5, 8))

    # add figures
    ax_objs = []
    for i in range(columns):
        # add ax
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # line plot
        plot = df.iloc[:, i].plot(color='k', lw=1.2)
        ax_objs[-1].yaxis.set_label_position("right")
        ax_objs[-1].set_ylabel(column_names[i], labelpad=15,
                               rotation=0, va='center', ha='left', ma='left')

        # area
        x = plot.get_children()[0]._x
        y = plot.get_children()[0]._y
        ax_objs[-1].fill_between(x, y, alpha=0.2,
                                 color=cmap(df.iloc[:, i]))
                                 #color=cmap(i))
        # ax_objs[-1].set_ylim(-2, 2)  # should be set from min/max
        rect = ax_objs[-1].patch
        # rect.set_alpha(0.2)

        # remove y ticks and labels
        # ax_objs[-1].set_yticklabels([])

        # conditionally remove other things
        if i != columns - 1:        # if not bottom plot
            # remove x-axis ticks and labels
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].set_xlabel('')
            # remove frame
            ax_objs[-1].tick_params(left=True,
                                    bottom=False)
            spines = ['top', 'right', 'left', 'bottom']
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)

        else:
            # remove only ticks
            ax_objs[-1].tick_params(left=True)
            # remove frames and keep only bottom
            spines = ['top', 'right', 'left']
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)

    # gs.update(hspace= -.06)
    # fig.set_tight_layout(True)
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.' + muscle + '.' + segment + '.' + neuron +
                    '.grid.pdf', bbox_inches='tight')


def polygon_under_graph(xlist, ylist):
    """Construct the vertex list which defines the polygon filling the
    space under the (xlist, ylist) line graph.  Assumes the xs are in
    ascending order.

    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


def plot_neuron_activity_3D(state, muscle, segment, neuron, cycle,
                            output_file=None):
    """Visualize neuron input activity using a 3D plot."""

    # columns and color map
    df = extract_neuron_activity(state, muscle, segment, neuron, cycle)
    columns = df.shape[1]
    column_names = df.columns.tolist()
    cmap = get_cmap(columns, 'jet')

    # The ith polygon will appear on the plane y = zs[i]
    zs = range(columns)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')

    verts = []
    for i in zs:
        data = df.iloc[:, i]
        x = data.index.tolist()
        y = data.tolist()
        # Make verts a list such that verts[i] is a list of (x, y) pairs
        # defining polygon i.
        verts.append(polygon_under_graph(x, y))

    poly = PolyCollection(verts, facecolors=[cmap(i) for i in
                                             range(columns)], alpha=.6)
    ax.add_collection3d(poly, zs=np.array(zs), zdir='y')

    ax.set_xlabel('gait cycle (%)')
    ax.set_zlabel('activity')
    ax.set_xlim(0, 100)
    ax.set_zlim(-1, 1)
    ax.set_ylim(0, columns)
    ax.set_yticks(range(columns))
    ax.set_yticklabels(column_names, fontsize=10, va='bottom', ha='left')
    # fig.set_tight_layout(True)
    # fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file + '.' + muscle + '.' + segment + '.' + neuron +
                    '.3D.pdf', bbox_inches='tight')


def plot_segment_chrod(state, segment, side, backend='matplotlib',
                       output_file=None):
    """"""
    # select columns
    regex = '.*_{side}.*{segment}.*->.*_{side}.*{segment}.*'\
        .format(side=side, segment=segment)
    mask = state.columns.str.contains(regex)
    df = state.iloc[:, mask]

    # transform data
    tran_df = pd.DataFrame(columns=['source', 'destination', 'strength'])
    for column in df.columns:
        split = column.split('->')
        source = split[0]
        destination = split[1]
        strength = np.abs(df[column].median())
        tran_df = tran_df.append({'source': source, 'destination': destination,
                                  'strength': strength}, ignore_index=True)

    # normalize strength
    max_lines = 50
    min_lines = 10
    min_s = tran_df.strength.min()
    max_s = tran_df.strength.max()
    tran_df.strength = (tran_df.strength - min_s) / (max_s - min_s) * max_lines
    tran_df.strength = tran_df.strength.astype(int)
    tran_df.strength = tran_df.strength + min_lines
    tran_df.sort_values(by=['strength'], ascending=True, inplace=True)

    # labels (it is important to sort them)
    neurons = list(set(tran_df['source'].unique().tolist() + \
                       tran_df['destination'].unique().tolist()))
    neurons.sort()
    neurons_dataset = hv.Dataset(pd.DataFrame(neurons, columns=['neuron']))

    # construct chord plot
    # hv.extension("bokeh")
    # hv.extension('matplotlib')
    hv.extension(backend)
    hv.output(size=300)
    chord = hv.Chord((tran_df, neurons_dataset), ['source', 'destination'],
                     ['strength'])
    chord.opts(title=segment, labels='neuron', node_color='neuron',
               node_cmap='Category20', edge_color='source',
               edge_cmap='Category20', edge_alpha=1.0)

    if output_file is not None:
        if backend == 'matplotlib':
            extension = '.pdf'
        else:
            extension = '.html'
        hv.save(chord, output_file + '.' + segment + '.chrod' + extension)
