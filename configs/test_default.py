import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    
    config.agentfile = './results/rllib/MuscleWalkingImitation2D-v0/PPO_2021-07-15_15-16-12/PPO_MuscleWalkingImitation2D-v0_7d3f0_00000_0_2021-07-15_15-16-12'
    config.checkpoint = 227
    config.mode = 'test'
    config.visualize = True
    config.apply_perturbations = True

    config.save_plots = True

    config.max_iterations = 10
    

    return config