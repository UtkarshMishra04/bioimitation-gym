from gym.envs.registration import register
import ray
from ray.tune.registry import register_env


########################################################################
#   Torque based environments and tasks
########################################################################

from bioimitation.imitation_envs.envs.torque.planar.torque_walking_imitation_env2D import TorqueWalkingImitationEnv2D
from bioimitation.imitation_envs.envs.torque.planar.torque_running_imitation_env2D import TorqueRunningImitationEnv2D
from bioimitation.imitation_envs.envs.torque.planar.torque_jumping_imitation_env2D import TorqueJumpingImitationEnv2D
from bioimitation.imitation_envs.envs.torque.planar.torque_locked_knee_imitation_env2D import TorqueLockedKneeImitationEnv2D
from bioimitation.imitation_envs.envs.torque.spatial.torque_walking_imitation_env3D import TorqueWalkingImitationEnv3D
from bioimitation.imitation_envs.envs.torque.spatial.torque_running_imitation_env3D import TorqueRunningImitationEnv3D
from bioimitation.imitation_envs.envs.torque.spatial.torque_jumping_imitation_env3D import TorqueJumpingImitationEnv3D
from bioimitation.imitation_envs.envs.torque.spatial.torque_locked_knee_imitation_env3D import TorqueLockedKneeImitationEnv3D

"""
Imitation environments
"""

register(
    id='TorqueWalkingImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.planar.torque_walking_imitation_env2D:TorqueWalkingImitationEnv2D',
)

register(
    id='TorqueRunningImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.planar.torque_running_imitation_env2D:TorqueRunningImitationEnv2D',
)

register(
    id='TorqueJumpingImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.planar.torque_jumping_imitation_env2D:TorqueJumpingImitationEnv2D',
)

register(
    id='TorqueLockedKneeImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.planar.torque_locked_knee_imitation_env2D:TorqueLockedKneeImitationEnv2D',
)

register(
    id='TorqueWalkingImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.spatial.torque_walking_imitation_env3D:TorqueWalkingImitationEnv3D',
)

register(
    id='TorqueRunningImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.spatial.torque_running_imitation_env3D:TorqueRunningImitationEnv3D',
)

register(
    id='TorqueJumpingImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.spatial.torque_jumping_imitation_env3D:TorqueJumpingImitationEnv3D',
)

register(
    id='TorqueLockedKneeImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.torque.spatial.torque_locked_knee_imitation_env3D:TorqueLockedKneeImitationEnv3D',
)

register_env('TorqueWalkingImitation2D-v0', lambda config: TorqueWalkingImitationEnv2D(config))
register_env('TorqueRunningImitation2D-v0', lambda config: TorqueRunningImitationEnv2D(config))
register_env('TorqueJumpingImitation2D-v0', lambda config: TorqueJumpingImitationEnv2D(config))
register_env('TorqueLockedKneeImitation2D-v0', lambda config: TorqueLockedKneeImitationEnv2D(config))
register_env('TorqueWalkingImitation3D-v0', lambda config: TorqueWalkingImitationEnv3D(config))
register_env('TorqueRunningImitation3D-v0', lambda config: TorqueRunningImitationEnv3D(config))
register_env('TorqueJumpingImitation3D-v0', lambda config: TorqueJumpingImitationEnv3D(config))
register_env('TorqueLockedKneeImitation3D-v0', lambda config: TorqueLockedKneeImitationEnv3D(config))

########################################################################
#   Muscle based environments and tasks
########################################################################

from bioimitation.imitation_envs.envs.muscle.planar.muscle_walking_imitation_env2D import MuscleWalkingImitationEnv2D
from bioimitation.imitation_envs.envs.muscle.planar.muscle_running_imitation_env2D import MuscleRunningImitationEnv2D
from bioimitation.imitation_envs.envs.muscle.planar.muscle_jumping_imitation_env2D import MuscleJumpingImitationEnv2D
from bioimitation.imitation_envs.envs.muscle.planar.muscle_locked_knee_imitation_env2D import MuscleLockedKneeImitationEnv2D
from bioimitation.imitation_envs.envs.muscle.spatial.muscle_walking_imitation_env3D import MuscleWalkingImitationEnv3D
from bioimitation.imitation_envs.envs.muscle.spatial.muscle_running_imitation_env3D import MuscleRunningImitationEnv3D
from bioimitation.imitation_envs.envs.muscle.spatial.muscle_jumping_imitation_env3D import MuscleJumpingImitationEnv3D
from bioimitation.imitation_envs.envs.muscle.spatial.muscle_locked_knee_imitation_env3D import MuscleLockedKneeImitationEnv3D
from bioimitation.imitation_envs.envs.muscle.spatial.muscle_palsy_imitation_env3D import MusclePalsyImitationEnv3D

"""
Imitation environments
"""

register(
    id='MuscleWalkingImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.planar.muscle_walking_imitation_env2D:MuscleWalkingImitationEnv2D',
)

register(
    id='MuscleRunningImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.planar.muscle_running_imitation_env2D:MuscleRunningImitationEnv2D',
)

register(
    id='MuscleJumpingImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.planar.muscle_jumping_imitation_env2D:MuscleJumpingImitationEnv2D',
)

register(
    id='MuscleLockedKneeImitation2D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.planar.muscle_locked_knee_imitation_env2D:MuscleLockedKneeImitationEnv2D',
)

register(
    id='MuscleWalkingImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.spatial.muscle_walking_imitation_env3D:MuscleWalkingImitationEnv3D',
)

register(
    id='MuscleRunningImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.spatial.muscle_running_imitation_env3D:MuscleRunningImitationEnv3D',
)

register(
    id='MuscleJumpingImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.spatial.muscle_jumping_imitation_env3D:MuscleJumpingImitationEnv3D',
)

register(
    id='MuscleLockedKneeImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.spatial.muscle_locked_knee_imitation_env3D:MuscleLockedKneeImitationEnv3D',
)

register(
    id='MusclePalsyImitation3D-v0',
    entry_point='bioimitation.imitation_envs.envs.muscle.spatial.muscle_palsy_imitation_env3D:MusclePalsyImitationEnv3D',
)

register_env('MuscleWalkingImitation2D-v0', lambda config: MuscleWalkingImitationEnv2D(config))
register_env('MuscleRunningImitation2D-v0', lambda config: MuscleRunningImitationEnv2D(config))
register_env('MuscleJumpingImitation2D-v0', lambda config: MuscleJumpingImitationEnv2D(config))
register_env('MuscleLockedKneeImitation2D-v0', lambda config: MuscleLockedKneeImitationEnv2D(config))
register_env('MuscleWalkingImitation3D-v0', lambda config: MuscleWalkingImitationEnv3D(config))
register_env('MuscleRunningImitation3D-v0', lambda config: MuscleRunningImitationEnv3D(config))
register_env('MuscleJumpingImitation3D-v0', lambda config: MuscleJumpingImitationEnv3D(config))
register_env('MuscleLockedKneeImitation3D-v0', lambda config: MuscleLockedKneeImitationEnv3D(config))
register_env('MusclePalsyImitation3D-v0', lambda config: MusclePalsyImitationEnv3D(config))