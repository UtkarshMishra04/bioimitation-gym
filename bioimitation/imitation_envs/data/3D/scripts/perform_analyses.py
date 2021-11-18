# Automates the execution of OpenSim analyses.
#
# It is better to execute the analyses one by one in the OpenSim GUI
# and perform the necessary sanity checks to be sure. Once we are sure
# that all stages work properly (e.g., scaling), this script can be
# used to automate the process. Make sure that the initial and final
# time of inverse dynamics and static optimization algorithm
# corresponds to when the legs touch the force plates. To check this,
# plot the vertical force of the left and right legs (e.g.,
# experimental_data/walk_grf.mot) to identify the time interval.
#
# author: Dimitar Stanev <jimstanev@gmail.com>
# %%
import os
from subprocess import call
from utils import subject_specific_isometric_force
from utils import plot_sto_file
# from utils import replace_thelen_muscles_with_millard

# %%
# subject data

# 1.61m, 41.5 kg
subject_height = 1.61
generic_height = 1.70
subject_dir = os.path.abspath('../')
os.chdir(subject_dir)

# switches
adapt_muscle_strength = False

# %%
# scale

os.chdir('scale/')

call(['opensim-cmd', 'run-tool', 'setup_scale.xml'])

# adjust max isometric force of subject-specific model based on height and
# weight regression model
if adapt_muscle_strength:
    subject_specific_isometric_force('../model/model_generic.osim',
                                     'model_scaled.osim',
                                     generic_height,
                                     subject_height)

os.chdir(subject_dir)

# %%
# inverse kinematics

os.chdir('inverse_kinematics/')

call(['opensim-cmd', 'run-tool', 'setup_ik.xml'])

plot_sto_file('task_InverseKinematics.mot', 'task_InverseKinematics.pdf', 3)
plot_sto_file('task_ik_marker_errors.sto', 'task_ik_marker_errors.pdf', 3)
plot_sto_file('task_ik_model_marker_locations.sto',
              'task_ik_model_marker_locations.pdf', 3)

os.chdir(subject_dir)

# %%
# kinematics analysis

os.chdir('kinematics_analysis/')

call(['opensim-cmd', 'run-tool', 'setup_ka.xml'])

os.chdir(subject_dir)

# %%
# muscle analysis

# os.chdir('muscle_analysis/')

# replace_thelen_muscles_with_millard('../scale/model_scaled.osim', '.')

# call(['opensim-cmd', 'run-tool', 'setup_ma.xml'])

# plot_sto_file('task_MuscleAnalysis_NormalizedFiberLength.sto',
#               'task_MuscleAnalysis_NormalizedFiberLength.pdf', 3)
# plot_sto_file('task_MuscleAnalysis_NormFiberVelocity.sto',
#               'task_MuscleAnalysis_NormFiberVelocity.pdf', 3)

# os.chdir(subject_dir)

# %%
# inverse dynamics

os.chdir('inverse_dynamics/')

call(['opensim-cmd', 'run-tool', 'setup_id.xml'])

plot_sto_file('task_InverseDynamics.sto', 'task_InverseDynamics.pdf', 3)

os.chdir(subject_dir)

# %%
# static optimization

os.chdir('static_optimization/')

call(['opensim-cmd', 'run-tool', 'setup_so.xml'])

plot_sto_file('task_StaticOptimization_activation.sto',
              'task_StaticOptimization_activation.pdf', 3)
plot_sto_file('task_StaticOptimization_force.sto',
              'task_StaticOptimization_force.pdf', 3)

os.chdir(subject_dir)

# %%
