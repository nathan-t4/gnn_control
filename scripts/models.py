import jraph
import jax
import diffrax
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

from typing import Sequence, Callable
from ml_collections import FrozenConfigDict
from utils.graph_utils import *

class MLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: str = 'swish'
    dropout_rate: float = 0
    deterministic: bool = True
    with_layer_norm: bool = False

    @nn.compact
    def __call__(self, inputs, training: bool=False):
        x = inputs
        if self.activation == 'swish':
            activation_fn = nn.swish
        elif self.activation == 'relu':
            activation_fn = nn.relu
        else:
            activation_fn = nn.softplus

        for i, size in enumerate(self.feature_sizes):
            x = nn.Dense(features=size)(x)
            if i != len(self.feature_sizes) - 1:
                x = activation_fn(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        if self.with_layer_norm:
            x = nn.LayerNorm()(x)
        return x

class GraphNetworkSimulator(nn.Module):
    """ 
        EncodeProcessDecode GN 
    """
    system_params: FrozenConfigDict
    norm_stats: FrozenConfigDict

    num_mp_steps: int = 1
    layer_norm: bool = False
    use_edge_model: bool = False
    shared_params: bool = False
    vel_history: int = 1
    control_history: int = 1
    noise_std: float = 0.0003

    globals_output_size: int = 0
    edge_output_size: int = 1
    node_output_size: int = 1
    prediction: str = 'control'
    
    # MLP parameters
    latent_size: int = 16
    hidden_layers: int = 2
    activation: str = 'relu'
    dropout_rate: float = 0
    training: bool = True

    add_self_loops: bool = False
    add_undirected_edges: bool = False

    dt: float = 0.01

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, rng: jax.Array) -> jraph.GraphsTuple:
        # next_u = next_u[1::2] # get nonzero elements (even indices) corresponding to control input
        if self.training: 
            # TODO: add noise to state?
            pass 
            # # Add noise to current position (first node feature)
            # rng, pos_rng, u_rng = jax.random.split(rng, 3)
            # pos_noise = self.noise_std * jax.random.normal(pos_rng, (len(graph.nodes),))
            # new_nodes = jnp.column_stack((graph.nodes[:,0].T + pos_noise, graph.nodes[:,1:]))
            # graph = graph._replace(nodes=new_nodes)
            # # Add noise to control input at current time-step (next_u)
            # next_u_noise = self.noise_std * jax.random.normal(u_rng, (len(next_u),))
            # next_u = next_u + next_u_noise

        if self.prediction == 'control':
            num_nodes = len(graph.nodes) 
            cur_state = graph.nodes[:,:2]

        def update_node_fn(nodes, senders, receivers, globals_):
            node_feature_sizes = [self.latent_size] * self.hidden_layers
            if self.use_edge_model:
                input = jnp.concatenate((nodes, senders, receivers), axis=1)
            else:
                input = nodes
            model = MLP(feature_sizes=node_feature_sizes, 
                        activation=self.activation, 
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(input)

        def update_edge_fn(edges, senders, receivers, globals_):
            edge_feature_sizes = [self.latent_size] * self.hidden_layers
            # input = jnp.concatenate((edges, senders, receivers, globals_), axis=1)
            input = jnp.concatenate((edges, senders, receivers), axis=1)
            model = MLP(feature_sizes=edge_feature_sizes,
                        activation=self.activation,
                        dropout_rate=self.dropout_rate, 
                        deterministic=not self.training,
                        with_layer_norm=self.layer_norm)
            return model(input)
            
        def update_global_fn(nodes, edges, globals_):
            del nodes, edges
            return globals_ 
        
        # Encoder
        if not self.use_edge_model:
            embed_edge_encoder = None
        else:
            embed_edge_encoder=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation)
        
        encoder = jraph.GraphMapFeatures(
            embed_edge_fn=embed_edge_encoder,
            embed_node_fn=MLP(feature_sizes=[self.latent_size] * self.hidden_layers,
                              with_layer_norm=self.layer_norm, 
                              activation=self.activation),
        )
        
        # Processor
        if not self.use_edge_model:
            update_edge_fn = None

        if graph.globals is None:
            update_global_fn = None

        num_nets = self.num_mp_steps if not self.shared_params else 1
        processor_nets = []
        for _ in range(num_nets): # TODO replace with scan
            net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )
            processor_nets.append(net)

        if self.use_edge_model:
            embed_edge_decoder=MLP(
                feature_sizes=[self.latent_size] * self.hidden_layers + [self.edge_output_size], 
                activation=self.activation),
        else:
            embed_edge_decoder = None

        # Decoder
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=MLP(
                feature_sizes=[self.latent_size] * self.hidden_layers + [self.node_output_size], 
                activation=self.activation),
            embed_edge_fn=embed_edge_decoder,
        )

        def decoder_postprocessor(graph: jraph.GraphsTuple):
            next_nodes = None
            next_edges = None
           
            if self.prediction == 'control':
                next_control = graph.nodes
                next_control = self.norm_stats.control.mean + self.norm_stats.control.std * next_control

                def platoon_dynamics_function(state, control):
                    state = state.flatten()
                    # reconstructed_control = np.zeros(10)
                    control = control.squeeze()
                    reconstructed_control = jnp.array([0, control[0], 0, control[1], 0, control[2], 0, control[3], 0, control[4]])
                    
                    # np_rng = np.random.default_rng()
                    dt = self.system_params['dt']
                    alpha = self.system_params['alpha']
                    m = self.system_params['m']
                    noise_std = self.system_params['noise_std']

                    A_11 = jnp.array([[1, dt], [0, 1]])
                    A_ii = jnp.array([[1, dt], [alpha * dt / m, 1]]) # i = 2,3,4,5
                    A_ij = jnp.array([[0, 0], [-alpha * dt / m, 0]]) # i = 2,3,4,5; j = i-1
                    B_ii = jnp.array([[0, 0], [0, dt / m]]) # i = 1,2,3,4,5

                    zeros = jnp.zeros((2,2))

                    A = jnp.block([[A_11, zeros, zeros, zeros, zeros], 
                                   [A_ij, A_ii, zeros, zeros, zeros],
                                   [zeros, A_ij, A_ii, zeros, zeros],
                                   [zeros, zeros, A_ij, A_ii, zeros],
                                   [zeros, zeros, zeros, A_ij, A_ii]])
                    
                    B = jnp.block([[B_ii, zeros, zeros, zeros, zeros],
                                   [zeros, B_ii, zeros, zeros, zeros],
                                   [zeros, zeros, B_ii, zeros, zeros],
                                   [zeros, zeros, zeros, B_ii, zeros],
                                   [zeros, zeros, zeros, zeros, B_ii]])
                    
                    D = noise_std * jax.random.normal(rng,(5,))
                    w = dt / m * jnp.array([0, D[0], 0, D[1], 0, D[2], 0, D[3], 0, D[4]])
                    next_state = A @ state + B @ reconstructed_control + w

                    return next_state
                next_state = platoon_dynamics_function(cur_state, next_control).reshape(5,-1) # TODO
                next_nodes = jnp.concatenate((next_state, next_control), axis=-1)

            if self.use_edge_model:
                if self.add_undirected_edges:
                    next_edges = jnp.concatenate((next_edges, next_edges), axis=0)
                if self.add_self_loops:
                    next_edges = jnp.concatenate((next_edges, jnp.zeros((num_nodes, 1))), axis=0)
                graph = graph._replace(nodes=next_nodes, edges=next_edges)
            else:
                graph = graph._replace(nodes=next_nodes)   

            return graph

        # Encode features to latent space
        processed_graph = encoder(graph)
        prev_graph = processed_graph
        # Message passing
        for i in range(num_nets): 
            processed_graph = processor_nets[i](processed_graph)
            if self.use_edge_model:
                processed_graph = processed_graph._replace(nodes=processed_graph.nodes + prev_graph.nodes,
                                                        edges=processed_graph.edges + prev_graph.edges)
            else:
                processed_graph = processed_graph._replace(nodes=processed_graph.nodes + prev_graph.nodes)
            prev_graph = processed_graph

        processed_graph = decoder(processed_graph)

        # Decoder post-processor
        processed_graph = decoder_postprocessor(processed_graph)

        return processed_graph