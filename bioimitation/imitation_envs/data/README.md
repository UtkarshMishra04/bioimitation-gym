# Generic Models

This folder contains the following generic models:

- 2D: A model with 10 degrees of freedom (DoFs) and 18 muscles and can
  be used for sagittal plane motion. It also contains OpenSim analysis
  for healthy locomotion. *Use this model for sagittal plane movement
  only.*
- 3D: A model with 19 degrees of freedom (DoFs) and 92 muscles and can
  be used for 3D gait analysis.
- 3D_simple: A model with 14 degrees of freedom (DoFs) and 22 muscles and can be
  used for 3D gait analysis.

In case you expirience problems visualizing the model, make sure that the
geometry folder is next to the model (.osim file). For Linux I have created a
symbolic link to avoid having multiple instances of the folder.

# Subject-specific models

We used the generic model (gait1992, gait1422) and not the simplified model
(gait0918). We will generalized this pipeline for a model with less muscles in
the future so it can speedup the training. For each subject, we can have a
separate folder containing the relative data. We should make sure that the
structure is consistent. Therefore, you can also output the subject-related data
in the same folder under a specific folder. One can use the data from OpenSim
(e.g., kinematics, marker positions or even EMG) for imitation learning.

*Convention (please use lowercase letters and underscore instead of spaces)*

The following structure is followed (same order of execution):

- experimental_data: .c3d files and files exported by
  `scripts/c3d_to_opensim.py` converter script
- model: folder containing the baseline model
- scale: folder containing the marker set, OpenSim setup files and the
  subject-specific scaled model
- inverse_kinematics: OpenSim setup file and results of inverse kinematics
  (joint angles)
- kinematics_analysis: OpenSim setup file and results of kinematics
  (joint angles and velocities) and body kinematics (body position and
  orientations)
- inverse_dynamics: OpenSim setup files and results from the inverse dynamics
  tool (joint moments)
- static_optimization: OpenSim setup files and results from static optimization
  (muscle forces and activations)


