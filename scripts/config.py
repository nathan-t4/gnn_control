import optax
import ml_collections

def create_gnc_config(args) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.system_params = ml_collections.ConfigDict({
        'dt': 0.1, # [s]
        'm': 1500, # [kg]
        'alpha': 50, # [N/m]
        'noise_std': 1000, # changed from 10000
        'timesteps': 100,
    })
    config.training_params = ml_collections.ConfigDict({
        'net_name': 'GNS',
        'dir': args.dir,
        'seed': 0,
        'horizon': 20, # horizon for training
        'trial_name': 'gnc',
        'loss_function': 'supervised',
        'num_epochs': int(50),
        'min_epochs': int(50),
        'batch_size': 2,
        'log_every_steps': 1,
        'eval_every_steps': 10,
        'checkpoint_every_steps': 5,
        'clear_cache_every_steps': 10,
        'add_undirected_edges': False,
        'add_self_loops': True,
        'train_multi_trajectories': True,
    })
    config.optimizer_params = ml_collections.ConfigDict({
        'learning_rate': optax.exponential_decay(init_value=1e-3, transition_steps=5e2, decay_rate=0.1, end_value=1e-5),
    })
    config.net_params = ml_collections.ConfigDict({
        'prediction': 'control',
        # 'vel_history': 5,
        # 'control_history': 5,
        'num_mp_steps': 4, # must =1
        # 'noise_std': 0.0003,
        'latent_size': 16,
        'hidden_layers': 2,
        'activation': 'relu',
        'use_edge_model': False,
        'layer_norm': True,
        'shared_params': False,
        'dropout_rate': 0.5,
        'add_undirected_edges': config.training_params.add_undirected_edges,
        'add_self_loops': config.training_params.add_self_loops,
    })
    return config