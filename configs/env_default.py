import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.visualize = False
    config.max_actuation = 200
    config.mode = 'train'
    config.log = False
    config.r_weights = [0.8, 0.2, 0.1]
    config.apply_perturbations = False
    config.use_target_obs = True
    config.use_GRF = True
    config.horizon = 5

    return config