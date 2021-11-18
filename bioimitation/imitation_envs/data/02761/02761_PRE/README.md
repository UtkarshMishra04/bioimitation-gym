# OpenSim template folder

## Subject description

The subject data represent a cerebral palsy (CP) male individual.

- Height: 1.43m
- Mass: 35kg

## Processing through OpenSim

In order to apply this folder on a new subject the following steps
should be followed. The setup files and results are organized and
stored in separate folders. After application of the OpenSim analyses
and the verification of results through manual inspection, the user
can use the `perform_analysis.py` script located in the `scripts`
folder to automate the analysis process.

### Data preparation (C3D conversation)

*This section can be skipped in case that you have the .trc (static
and dynamic trial) and .mot (ground reaction forces) files. However,
please follow the same naming convention so that scripts can be used
seamlessly.*

1. Copy the static and task (e.g., walking) trials into
   `experimenta_data` folder. Rename the files as `static.c3d` and
   `task.c3d`. The name convention is important for the scripts.*Make
   sure to check the labels for the ground reaction force and make the
   necessary changes so that left and right legs are properly
   recognized.*
2. Execute the `c3d_to_opensim.py` script located in the `scripts`
   folder. With this script, we transform the lab's coordinate system
   into OpenSim's compatible coordinate system. We also change the
   units of the c3d file so that they are in F, m, and Nm. We
   interpolate the raw data to fill in missing values and we also
   filter the data. It is important to inspect the generated plots for
   any artifacts that could compromise the OpenSim analysis (requires
   expertise). Finally, note that if this template folder is used with
   a different lab coordinate system, then the transformations must be
   redefined.
3. Execute the `extract_emg_from_c3d.py` script located in the
   `scripts` folder. In this script we assume that we know the names
   of the muscles. One can inspect a .c3d file with Mokka and get the
   names of the muscles manually.

### Scaling in OpenSim

*In case that you use a different marker set, please update the marker
set and the scaling setup files accordingly.*

The Sinergia data set uses the minimum set of markers in the
conventional gait model. The user must specify the subject's mass,
height, age and notes in the `setup_scale.xml` located in the `scale`
folder. The one can do the scaling through the GUI and inspect the
results. It is advised that the user performs this operation through
the GUI and do not use the automating script
`perform_analysis.py`. Explanations of how to improve scaling is out
of the scope of this document and it requires a lot of experience.

### Inverse Kinematics

*In case that you use a different marker set, then please update the
tracking tasks in the `setup_ik.xml` file.*

All the setup files are located in the `inverse_kinematics` folder.

### Inverse dynamics

All the setup files are located in the `inverse_dynamics` folder.

### Static optimization

All the setup files are located in the `static_optimization` folder.
