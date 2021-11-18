import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-4
    config.value_lr = 1e-4
    config.critic_lr = 1e-4
    config.temp_lr = 1e-4

    config.hidden_dims = (512, 512)

    config.discount = 0.995

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.batch_size = 3200
    config.replay_buffer_size = int(1e6)

    config.log_interval = int(1e3)
    config.eval_interval = int(5e3)
    config.eval_episodes = 10

    config.start_training = int(1e4)

    return config
