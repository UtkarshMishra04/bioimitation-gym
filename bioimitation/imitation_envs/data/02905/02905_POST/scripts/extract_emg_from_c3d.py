# Script for processing EMG data from .c3d files.
#
# BTK should be installed either through conda on Windows:
#
#    conda install -c conda-forge btk
#
# or build from source:
#
#   https://github.com/mitkof6/BTKCore/tree/btk_python_setup
#
# author: Dimitar Stanev <jimstanev@gmail.com>
##
import os
import sys
sys.path.append('/home/utkubuntu/opensim/opensim_mycode/BTKCore/build/btk/')
import btk
import numpy as np
from utils import calculate_emg_linear_envelope, plot_sto_file
from utils import np_array_to_simtk_matrix, create_opensim_storage

##
# c3d

input_dir = os.path.abspath('../experimental_data/')
c3d_file_path = os.path.join(input_dir, 'task.c3d')
output_dir = input_dir

# read c3d file
c3d = btk.btkAcquisitionFileReader()
c3d.SetFilename(c3d_file_path)
c3d.Update()

# get analog data
acq = c3d.GetOutput()
f_s = acq.GetAnalogFrequency()
N = acq.GetAnalogFrameNumber()

##
# extract EMG raw data and compute envelope

muscles = ['RF', 'ST', 'TA', 'GL', 'PE']

time = np.linspace(0, float(N) / f_s, N, endpoint=True)
emg_raw = []
emg_env = []
labels = []
for muscle in muscles:
    # right leg
    labels.append('R_' + muscle)
    right = acq.GetAnalog('R_' + muscle).GetData().GetValues().reshape(-1)
    emg_raw.append(right)
    emg_env.append(calculate_emg_linear_envelope(right, f_s))
    # left leg
    labels.append('L_' + muscle)
    left = acq.GetAnalog('L_' + muscle).GetData().GetValues().reshape(-1)
    emg_raw.append(left)
    emg_env.append(calculate_emg_linear_envelope(left, f_s))

##
# save data

emg_raw_sto = create_opensim_storage(time,
                                     np_array_to_simtk_matrix(
                                         np.array(emg_raw).transpose()),
                                     labels)
emg_raw_sto.setName('emg_raw')
emg_raw_sto.printResult(emg_raw_sto, 'emg_raw', output_dir, 1.0 / f_s, '.sto')
# plot_sto_file(os.path.join(output_dir, 'emg_raw.sto'),
#               os.path.join(output_dir, 'emg_raw.pdf'), 2)

emg_env_sto = create_opensim_storage(time,
                                     np_array_to_simtk_matrix(
                                         np.array(emg_env).transpose()),
                                     labels)
emg_env_sto.setName('emg_env')
emg_env_sto.printResult(emg_env_sto, 'emg_env', output_dir, 1.0 / f_s, '.sto')
# plot_sto_file(os.path.join(output_dir, 'emg_env.sto'),
#               os.path.join(output_dir, 'emg_env.pdf'), 2)


##
