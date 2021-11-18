# Brief Visualize kinematics, kinematics, and muscle activity of a SCONE
# simulation.
#
# author: Dimitar Stanev <dimitar.stanev@epfl.ch>
# %%
import os
from bioimitation.imitation_envs.utils.visualization_utils2D import read_from_storage
from bioimitation.imitation_envs.utils.visualization_utils2D import perform_muscle_analysis
from bioimitation.imitation_envs.utils.visualization_utils2D import plot_scone_kinematics
from bioimitation.imitation_envs.utils.visualization_utils2D import plot_scone_vertical_reactions
from bioimitation.imitation_envs.utils.visualization_utils2D import plot_scone_joint_moments
from bioimitation.imitation_envs.utils.visualization_utils2D import plot_scone_muscle_excitations


# %%
# settings
def visualize_performance(save_path):

    model_file = os.path.abspath('./bioimitation/imitation_envs/data/2D/scale/model_scaled.osim')
    state_file = os.path.abspath(save_path+'simulation_States.sto')
    grf_file = os.path.abspath(save_path+'simulation_ForceReporter_forces.sto')
    state = read_from_storage(state_file)
    grf = read_from_storage(grf_file)
    muscle_analysis_output_dir = os.path.abspath(save_path+'muscle_analysis/')

    side = 'r'
    muscles = ['hamstrings', 'glut_max', 'iliopsoas', 'vasti', 'gastroc', 'soleus','tib_ant']
    columns = 4

    # %%
    # perform muscle analysis to calculate muscle induced moments

    # run this once
    if not os.path.isdir(muscle_analysis_output_dir):
        perform_muscle_analysis(model_file, state_file, muscle_analysis_output_dir)

    # %%
    # plot joint moments

    plot_scone_kinematics(state, grf, side, state_file)
    plot_scone_vertical_reactions(state, grf, side, model_file, state_file)
    plot_scone_joint_moments(state, grf, model_file, muscle_analysis_output_dir, side,
                            state_file)
    plot_scone_muscle_excitations(state, grf, muscles, side, columns, state_file)

    # %%
