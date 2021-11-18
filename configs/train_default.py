import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.env = None
    config.env_config = None
    config.model = {
        "fcnet_hiddens": [512, 512],
        "fcnet_activation": "tanh",
        "max_seq_len": 0
    }

    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
    config.lr = 1e-4  # try different lrs with grid search i.e. [1e-3, 1e-4, 1e-5]
    config.framework = "torch"
    config.gamma = 0.995
    config.kl_coeff = 1.0
    config.num_sgd_iter = 20
    config.sgd_minibatch_size = 256
    config.horizon = 5000
    config.train_batch_size = 3200
    config.num_workers = 1
    config.num_gpus = 0
    config.batch_mode = "complete_episodes"
    config.observation_filter = "MeanStdFilter"

    return config