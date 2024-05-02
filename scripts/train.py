import os
import jax
import optax
import json

import matplotlib.pyplot as plt
import jax.numpy as jnp

import ml_collections
from clu import metric_writers
from clu import periodic_actions
from clu import checkpoint

from flax.training.train_state import TrainState
from flax.training.early_stopping import EarlyStopping

from time import strftime
from timeit import default_timer
from graph_builder import VehiclePlatoonGraphBuilder
from scripts.models import *
from utils.data_utils import *
from utils.jax_utils import *
from utils.gnn_utils import *

def train(config: ml_collections.ConfigDict):
    system_params = config.system_params
    training_params = config.training_params
    net_params = config.net_params
    net_params.system_params = system_params

    def create_net(params):
        return GraphNetworkSimulator(**params)
    
    if training_params.dir == None:
        training_params.dir = os.path.join(os.curdir, f'results/gnc/{training_params.loss_function}_{strftime("%m%d%H%M")}')
            
    log_dir = os.path.join(training_params.dir, 'log')
    checkpoint_dir = os.path.join(training_params.dir, 'checkpoint')
    plot_dir = os.path.join(training_params.dir, 'plots')

    key = jax.random.key(training_params.seed)
    key, init_rng, net_rng = jax.random.split(key, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)
    # Create optimizer
    tx = optax.adam(**config.optimizer_params)
    # Create training and evaluation data loaders
    gb = VehiclePlatoonGraphBuilder(None, 
                                    training_params.seed,
                                    system_params.alpha, 
                                    system_params.dt,
                                    system_params.m,
                                    system_params.noise_std,
                                    int(1e5),
                                    jnp.array([0,1,2,3]),
                                    jnp.array([1,2,3,4]))  
    
    # TODO: make eval initial states out-of-distribution?
    eval_gb = VehiclePlatoonGraphBuilder(None, 
                                         training_params.seed,
                                         system_params.alpha, 
                                         system_params.dt,
                                         system_params.m,
                                         system_params.noise_std,
                                         20, 
                                         jnp.array([0,1,2,3]),
                                         jnp.array([1,2,3,4]))    
    
    # print('control mean', gb.norm_stats.control.mean, 'control std', gb.norm_stats.control.std)
    # Initialize training network
    net_params.norm_stats = gb.norm_stats
    net = create_net(net_params)
    init_state = gb.initial_states[0]
    init_graph = gb.get_graph(init_state)
    params = net.init(init_rng, init_graph, net_rng)
    batched_apply = jax.vmap(net.apply, in_axes=(None,0,None))

    print(f"Number of parameters {num_parameters(params)}")

    def random_batch(batch_size: int, min: int, max: int, key: jax.Array):
        """ Return random permutation of jnp.arange(min, max) in batches of batch_size """
        steps_per_epoch = (max - min)// batch_size
        perms = jax.random.permutation(key, max - min)
        perms = perms[: steps_per_epoch * batch_size].reshape(-1,batch_size)
        return perms
    
    def train_epoch(state: TrainState, batch_size: int, key: jax.Array):
        '''
            - set a finite horizon. vary x0 (aka the graphs) by batches -> run for finite horizon -> calculate J
        '''     
        loss_function = training_params.loss_function
        x0_perms = random_batch(batch_size, 0, len(gb.initial_states), key) 
        dropout_rng = jax.random.split(key, batch_size)
        key, net_rng = jax.random.split(key)

        def loss_fn(params, batch_x0s, batch_graphs):
            if loss_function == 'quadratic_loss':
                def apply_fn(graph, _):
                    pred_graph = state.apply_fn(params, graph, net_rng, rngs={'dropout': dropout_rng})
                    state_predictions = pred_graph.nodes[:,:,:2].reshape((training_params.batch_size, gb.state_dim))
                    control_predictions = pred_graph.nodes[:,:,-1]
                    loss = jnp.sum(state_predictions @ gb.Q @ state_predictions.T   \
                                 + control_predictions @ gb.R @ control_predictions.T)
                    pred_graph = pred_graph._replace(nodes=pred_graph.nodes[:,:,:-1])
                    return pred_graph, loss
                final_graph, losses = jax.lax.scan(apply_fn, batch_graphs, None, length=training_params.horizon)
                loss = jnp.sum(losses)
            elif loss_function == 'supervised':
                def apply_fn(graph, control_targets):
                    def multi_apply(graph, _):
                        pred_graph = state.apply_fn(params, graph, net_rng, rngs={'dropout': dropout_rng})
                        control_predictions = pred_graph.nodes[:,:,-1].reshape((1, gb.state_dim))
                        pred_graph = pred_graph._replace(nodes=pred_graph.nodes[:,:,:-1])
                        return pred_graph, control_predictions
                    pred_graph, control_predictions = jax.lax.scan(multi_apply, graph, None, length=training_params.horizon)
                    control_predictions = jnp.array(control_predictions).reshape((training_params.horizon, gb.state_dim))
                    loss = optax.l2_loss(control_predictions, control_targets)
                    return pred_graph, loss
            
                subkeys = jax.random.split(key)
                get_optimal_state_trajectory = jax.vmap(gb.get_optimal_state_trajectory, in_axes=(0,0,None))
                state_targets, control_targets = get_optimal_state_trajectory(subkeys, batch_x0s, training_params.horizon)
                final_graph, losses = jax.lax.scan(apply_fn, batch_graphs, control_targets)
                loss = jnp.sum(losses)
            return loss

        def train_batch(state, x0_idx):
            x0s = gb.initial_states[x0_idx]
            graphs = gb.get_graph_batch(x0s)
            loss, grads = jax.value_and_grad(loss_fn)(state.params, x0s, graphs)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        state, epoch_loss = jax.lax.scan(train_batch, state, x0_perms)

        train_loss = jnp.asarray(epoch_loss).mean()

        return state, TrainMetrics.single_from_model_output(loss=train_loss)

    def rollout(state: TrainState, x0: jnp.array, key: jax.Array):
        loss_function = training_params.loss_function
        num_ts = system_params.timesteps // net_params.num_mp_steps
        tf_idxs = jnp.arange(num_ts)
        ts = tf_idxs * system_params.dt

        state_targets, control_targets = gb.get_optimal_state_trajectory(key, x0, num_ts)
        graph0 = gb.get_graph(x0)
        # graph0 = pytrees_stack([graph0])
        keys = jax.random.split(key, num_ts)

        def forward_pass(graph, inputs):
            if loss_function == 'quadratic_loss':
                key, timestep = inputs
                graph = state.apply_fn(state.params, graph, key)
                # compute gnc loss
                state_predictions = (graph.nodes[:,:2]).reshape((10,-1))
                control_predictions = (graph.nodes[:,-1]).squeeze()
                # Graph features (state) is updated in the apply_fn
                graph = graph._replace(nodes=graph.nodes[:,:-1])
                loss = state_predictions.T @ gb.Q @ state_predictions \
                     + control_predictions.T @ gb.R @ control_predictions
                return graph, (state_predictions, control_predictions, loss)
            elif loss_function == 'supervised':
                key, timestep = inputs
                graph = state.apply_fn(state.params, graph, key)
                state_predictions = (graph.nodes[:,:2]).reshape((10,-1))
                control_predictions = (graph.nodes[:,-1]).squeeze()
                position_predictions = state_predictions[::2].squeeze()
                position_targets = state_targets.reshape((10,-1))[::2, timestep]
                graph = graph._replace(nodes=graph.nodes[:,:-1])
                loss = optax.l2_loss(position_predictions, position_targets)
                return graph, (state_predictions, control_predictions, loss)

        graphT, pred_data = jax.lax.scan(forward_pass, graph0, (keys, tf_idxs))

        exp_data = (state_targets, control_targets) # Optimal LQR state and control
        losses = pred_data[2]
        pred_data = pred_data
        total_loss = jnp.sum(losses)
        
        return ts, pred_data, exp_data, EvalMetrics.single_from_model_output(loss=total_loss)

    state = TrainState.create(
        apply_fn=batched_apply,
        params=params,
        tx=tx,
    )

    # Create evaluation network
    eval_net = create_net(net_params)
    eval_net.training = False
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=training_params.num_epochs,
        writer=writer
    )
    # Load previous checkpoint (if applicable)
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    best_model_ckpt = checkpoint.Checkpoint(os.path.join(checkpoint_dir, 'best_model'))
    state = best_model_ckpt.restore_or_initialize(state)

    steps_per_epoch = gb.num_initial_states // training_params.batch_size
    train_fn = train_epoch
    # Setup training epochs
    init_epoch = int(state.step) // steps_per_epoch + 1
    final_epoch = init_epoch + training_params.num_epochs
    training_params.num_epochs = final_epoch

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)

    train_metrics = None
    min_error = jnp.inf
    print("Start training")
    for epoch in range(init_epoch, final_epoch):
        key, train_rng = jax.random.split(key)
        state, metrics_update = train_fn(state, training_params.batch_size, train_rng) 
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)

        print(f'Epoch {epoch}: loss = {jnp.round(train_metrics.compute()["loss"], 4)}')

        is_last_step = (epoch == final_epoch - 1)

        if epoch % training_params.eval_every_steps == 0 or is_last_step:
            eval_metrics = None
            net.training = False
            with report_progress.timed('eval'):
                eval_state = eval_state.replace(params=state.params)
                rollout_error_sum = 0
                for i in range(len(eval_gb.initial_states)):
                    key, eval_key = jax.random.split(key)
                    ts, pred_data, exp_data, eval_metrics = rollout(eval_state, eval_gb.initial_states[i], eval_key)
                    rollout_error_sum += eval_metrics.compute()['loss']
                    plot_evaluation_curves(ts, pred_data, exp_data, None,
                                           plot_dir=os.path.join(plot_dir, f'traj_{i}'),
                                           prefix=f'Epoch {epoch}: eval_traj_{i}')
                
                mean_loss = rollout_error_sum / eval_gb.num_initial_states
                writer.write_scalars(epoch, add_prefix_to_keys({'loss': mean_loss}, 'eval'))
                print(f'Epoch {epoch}: rollout mean position loss = {jnp.round(mean_loss, 4)}')

                if mean_loss < min_error: 
                    # Save best model
                    min_error = mean_loss
                    with report_progress.timed('checkpoint'):
                        best_model_ckpt.save(state)
                if epoch > training_params['min_epochs']: # train at least for 'min_epochs' epochs
                    early_stop = early_stop.update(mean_loss)
                    if early_stop.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch}')
                        training_params.num_epochs = epoch - init_epoch
                        is_last_step = True
            net.training = True

        if epoch % training_params.log_every_steps == 0 or is_last_step:
            writer.write_scalars(epoch, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        if epoch % training_params.checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed('checkpoint'):
                ckpt.save(state)

        if epoch % training_params.clear_cache_every_steps == 0 or is_last_step: 
            jax.clear_caches()

        if is_last_step:
            break

    # Save config to json
    config_js = config.to_json_best_effort()
    run_params_file = os.path.join(training_params.dir, 'run_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(config_js, outfile)

def eval(config: ml_collections.ConfigDict):
    system_params = config.system_params
    training_params = config.training_params
    net_params = config.net_params
    net_params.system_params = system_params

    def create_net(params):
        return GraphNetworkSimulator(**params)
    
    if training_params.dir == None:
        training_params.dir = os.path.join(os.curdir, f'results/gnc/{training_params.loss_function}_{strftime("%m%d%H%M")}')
            
    log_dir = os.path.join(training_params.dir, 'log')
    checkpoint_dir = os.path.join(training_params.dir, 'checkpoint')
    plot_dir = os.path.join(training_params.dir, 'plots')
    eval_plot_dir = os.path.join(plot_dir, 'eval')

    key = jax.random.key(training_params.seed)
    key, init_rng, net_rng = jax.random.split(key, 3)

    # Create writer for logs
    writer = metric_writers.create_default_writer(logdir=log_dir)
    # Create optimizer
    tx = optax.adam(**config.optimizer_params)
    # Create training and evaluation data loaders
    gb = VehiclePlatoonGraphBuilder(None, 
                                    training_params.seed,
                                    system_params.alpha, 
                                    system_params.dt,
                                    system_params.m,
                                    system_params.noise_std,
                                    int(1e5),
                                    jnp.array([0,1,2,3]),
                                    jnp.array([1,2,3,4]))  
    
    # TODO: make eval initial states out-of-distribution?
    eval_gb = VehiclePlatoonGraphBuilder(None, 
                                         training_params.seed,
                                         system_params.alpha, 
                                         system_params.dt,
                                         system_params.m,
                                         system_params.noise_std,
                                         20, 
                                         jnp.array([0,1,2,3]),
                                         jnp.array([1,2,3,4]))    
    
    # print('control mean', gb.norm_stats.control.mean, 'control std', gb.norm_stats.control.std)
    # Initialize training network
    net_params.norm_stats = gb.norm_stats
    net = create_net(net_params)
    net.training = False
    init_state = eval_gb.initial_states[0]
    init_graph = eval_gb.get_graph(init_state)
    params = net.init(init_rng, init_graph, net_rng)

    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
    )

    # Load previous checkpoint (if applicable)
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    best_model_ckpt = checkpoint.Checkpoint(os.path.join(checkpoint_dir, 'best_model'))
    state = best_model_ckpt.restore_or_initialize(state)

    print(f"Number of parameters {num_parameters(params)}")


    def rollout(state: TrainState, x0: jnp.array, key: jax.Array):
        loss_function = training_params.loss_function
        num_ts = system_params.timesteps // net_params.num_mp_steps
        tf_idxs = jnp.arange(num_ts)
        ts = tf_idxs * system_params.dt

        state_targets, control_targets = gb.get_optimal_state_trajectory(key, x0, num_ts)
        optimal_loss = state_targets @ gb.Q @ state_targets.T \
                     + control_targets @ (0.1 * jnp.eye(gb.state_dim)) @ control_targets.T
        optimal_loss = jnp.sum(optimal_loss)

        graph0 = gb.get_graph(x0)
        # graph0 = pytrees_stack([graph0])
        keys = jax.random.split(key, num_ts)

        def forward_pass(graph, inputs):
            if loss_function == 'quadratic_loss':
                key, timestep = inputs
                graph = state.apply_fn(state.params, graph, key)
                # compute gnc loss
                state_predictions = (graph.nodes[:,:2]).reshape((10,-1))
                control_predictions = (graph.nodes[:,-1]).squeeze()
                # Graph features (state) is updated in the apply_fn
                graph = graph._replace(nodes=graph.nodes[:,:-1])
                loss = state_predictions.T @ gb.Q @ state_predictions \
                     + control_predictions.T @ gb.R @ control_predictions
                return graph, (state_predictions, control_predictions, loss)
            elif loss_function == 'supervised':
                key, timestep = inputs
                graph = state.apply_fn(state.params, graph, key)
                state_predictions = (graph.nodes[:,:2]).reshape((10,-1))
                control_predictions = (graph.nodes[:,-1]).squeeze()
                position_predictions = state_predictions[::2].squeeze()
                position_targets = state_targets.reshape((10,-1))[::2, timestep]
                graph = graph._replace(nodes=graph.nodes[:,:-1])
                loss = optax.l2_loss(position_predictions, position_targets)
                return graph, (state_predictions, control_predictions, loss)

        graphT, pred_data = jax.lax.scan(forward_pass, graph0, (keys, tf_idxs))

        exp_data = (state_targets, control_targets) # Optimal LQR state and control
        losses = pred_data[2]
        pred_data = pred_data
        total_loss = jnp.sum(losses)
        
        return ts, pred_data, exp_data, EvalMetrics.single_from_model_output(loss=total_loss), EvalMetrics.single_from_model_output(loss=optimal_loss)

    # # Create evaluation network
    # eval_net = create_net(net_params)
    # eval_net.training = False
    # eval_state = state.replace(apply_fn=eval_net.apply)

    # Create logger to report training progress
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=training_params.num_epochs,
        writer=writer
    )


    steps_per_epoch = gb.num_initial_states // training_params.batch_size
    # Setup training epochs
    init_epoch = int(state.step) // steps_per_epoch + 1
    final_epoch = init_epoch + training_params.num_epochs
    training_params.num_epochs = final_epoch


    print("Start evaluation")

    eval_metrics = None
    net.training = False
    rollout_error_sum = 0
    rollout_optimal_error_sum = 0
    for i in range(len(eval_gb.initial_states)):
        key, eval_key = jax.random.split(key)
        ts, pred_data, exp_data, eval_metrics, optimal_loss = rollout(state, eval_gb.initial_states[i], eval_key)
        rollout_error_sum += eval_metrics.compute()['loss']
        rollout_optimal_error_sum += optimal_loss.compute()['loss']
        writer.write_scalars(i, add_prefix_to_keys({'loss': eval_metrics.compute()['loss']}, f'eval_{training_params.trial_name}'))
        plot_evaluation_curves(ts, pred_data, exp_data, None,
                                plot_dir=eval_plot_dir,
                                prefix=f'eval_traj_{i}')
    
    mean_loss = rollout_error_sum / eval_gb.num_initial_states
    mean_optimal_loss = rollout_optimal_error_sum / eval_gb.num_initial_states

    print(f'Mean rollout loss = {jnp.round(mean_loss, 4)}')
    print(f'Mean optimal loss = {jnp.round(mean_optimal_loss, 4)}')

    # Save config to json
    eval_config = ml_collections.ConfigDict({
        'mean_rollout_loss': mean_loss,
        'mean_optimal_loss': mean_optimal_loss,
        'system_params': system_params,
    })

    eval_config_js = eval_config.to_json_best_effort()
    run_params_file = os.path.join(training_params.dir, f'eval_{training_params.trial_name}_params.js')
    with open(run_params_file, "w") as outfile:
        json.dump(eval_config_js, outfile)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from scripts.config import create_gnc_config

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    cfg = create_gnc_config(args)

    if args.eval:
        eval(cfg)
    else:
        train(cfg)