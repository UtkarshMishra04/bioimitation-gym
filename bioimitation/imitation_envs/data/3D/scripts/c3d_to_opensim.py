# Conversation of .c3d files to OpenSim marker.trc and ground reaction forces
# grf.mot for the Sinergia data set. This script can be used for other data sets
# as well, however, the column names and transformation conventions may be
# different. Also, note that here we do not distinguish between left and right
# foot, therefore the setup_grf.xml file has to be manually updated.
#
# author: Dimitar Stanev <jimstanev@gmail.com>
# contributors: Celine Provins, George Papoulias
# %%
import os
import opensim
from utils import rotate_data_table, mm_to_m, plot_sto_file
from utils import refine_ground_reaction_wrench, create_opensim_storage

# %%
# specify the .c3d file

static_file = 'static.c3d'
task_file = 'task.c3d'
input_dir = os.path.abspath('../experimental_data/')
c3d_dir = input_dir
output_dir = input_dir

# define order for ground reaction wrench
labels_wrench = ['ground_force_vx', 'ground_force_vy', 'ground_force_vz',
                 'ground_force_px', 'ground_force_py', 'ground_force_pz',
                 'ground_torque_x', 'ground_torque_y', 'ground_torque_z']
# !! Here we assume that force plate #1 is the left leg and force plate #2 is
# right. One can swap the prefix below after inspecting the c3d file with Mokka.
print('Warning: it is assumed that force plate #1 records left leg '
      'and force plate #2 right. Please change accordingly (swap prefix).')
labels_force = ['left_'  + label for label in labels_wrench] +\
               ['right_' + label for label in labels_wrench]

# OpenSim data adapters
adapter = opensim.C3DFileAdapter()
adapter.setLocationForForceExpression(
    opensim.C3DFileAdapter.ForceLocation_CenterOfPressure)
trc_adapter = opensim.TRCFileAdapter()

# %%
# extract data for static trial

# get markers
static = adapter.read(os.path.join(c3d_dir, static_file))
markers_static = adapter.getMarkersTable(static)

# process markers and save to .trc file
rotate_data_table(markers_static, [1, 0, 0], -90)
trc_adapter.write(markers_static, os.path.join(output_dir, 'static.trc'))

# %%
# extract data for task (e.g., walk, run)

# get markers and forces
task = adapter.read(os.path.join(c3d_dir, task_file))
markers_task = adapter.getMarkersTable(task)
forces_task = adapter.getForcesTable(task)

# process markers of task and save to .trc file
rotate_data_table(markers_task, [1, 0, 0], -90)
trc_adapter = opensim.TRCFileAdapter()
trc_adapter.write(markers_task, os.path.join(output_dir, 'task.trc'))

# process forces
rotate_data_table(forces_task, [1, 0, 0], -90)

# conversion of unit (f -> N, p -> mm, tau -> Nmm)
mm_to_m(forces_task, 'p1')
mm_to_m(forces_task, 'p2')
mm_to_m(forces_task, 'm1')
mm_to_m(forces_task, 'm2')

# refine ground reaction forces
refine_ground_reaction_wrench(forces_task, ['f1', 'p1', 'm1'],
                              stance_threshold=50, tau=0.001)
refine_ground_reaction_wrench(forces_task, ['f2', 'p2', 'm2'],
                              stance_threshold=50, tau=0.001)

# export forces (assume two force plates)
time = forces_task.getIndependentColumn()
forces_task = forces_task.flatten(['x', 'y', 'z'])
force_sto = create_opensim_storage(time, forces_task.getMatrix(), labels_force)
force_sto.setName('GRF')
force_sto.printResult(force_sto, 'task_grf', output_dir, 0.01, '.mot')

# plot
plot_sto_file(os.path.join(output_dir, 'task.trc'),
              os.path.join(output_dir, 'task.pdf'), 3)
plot_sto_file(os.path.join(output_dir, 'task_grf.mot'),
              os.path.join(output_dir, 'task_grf.pdf'), 3)

# %%
