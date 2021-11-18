#!/bin/bash
cd ..
find . -name '*.log' -delete
find . -name '*.sto' -delete
find . -name '*.pdf' -delete
find . -name '*adjusted*' -delete
find . -name '*~' -delete
find . -name '__pycache__' -type d -exec rm -r '{}' \;
# rm experimental_data/emg_raw.sto
# rm experimental_data/emg_env.sto
rm scale/model_scaled.osim
rm scale/static.mot
rm scale/model_scale_set_applied.xml
rm scale/model_markers.osim
rm scale/model_static.mot
rm inverse_kinematics/task_InverseKinematics.mot
rm static_optimization/task_StaticOptimization_controls.xml
cd scripts/
