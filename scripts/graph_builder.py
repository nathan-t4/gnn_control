import jraph
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from jax.tree_util import register_pytree_node_class
from typing import Sequence
from functools import partial
from control import dlqr
from utils.graph_utils import add_edges

class GraphBuilder():
    def __init__(self, path, add_undirected_edges, add_self_loops):
        self._path = path
        self._add_undirected_edges = add_undirected_edges
        self._add_self_loops = add_self_loops
        self._load_data(self._path)
        self._get_norm_stats()
        self._setup_graph_params()

    def init(**kwargs):
        raise NotImplementedError

    def _load_data(self, path):
        raise NotImplementedError

    def _get_norm_stats(self):
        raise NotImplementedError
    
    def _setup_graph_params():
        raise NotImplementedError
    
    def get_graph(self, **kwargs) -> jraph.GraphsTuple:
        raise NotImplementedError
    
    def get_graph_batch(self, **kwargs) -> Sequence[jraph.GraphsTuple]:
        raise NotImplementedError
    
    def tree_flatten():
        raise NotImplementedError
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        raise NotImplementedError
    
@register_pytree_node_class
class MSDGraphBuilder(GraphBuilder):
    """ 
        Double Mass Spring Damper (DMSD) 
    """
    def __init__(self, path, add_undirected_edges, add_self_loops, mode, vel_history, control_history):
        super().__init__(path, add_undirected_edges, add_self_loops)
        self._mode = mode
        self._vel_history = vel_history
        self._control_history = control_history
    
    def _load_data(self, path):
        """
            The resulting dataset has dimensions [num_trajectories, num_timesteps, (qs, dqs, ps, accs)]

        """
        data = np.load(path, allow_pickle=True)
        state = data['state_trajectories']
        config = data['config']
        control = data['control_inputs']
        self._dt = config['dt']
        # Control
        # self._control = jnp.concatenate((jnp.zeros(control.shape), control), axis=-1)
        self._control = jnp.array(control)
        # Masses
        # self._m = jnp.array([config['m1'], config['m2']]).T
        self._m = jnp.array(config['m'])
        # Spring constants
        # self._k = jnp.array([config['k1'], config['k2']]).T
        self._k = jnp.array(config['k'])
        # Damper constants
        # self._b = jnp.array([config['b1'], config['b2']]).T
        self._b = jnp.array(config['b'])
        # Absolute position
        self._qs = jnp.array(state[:,:,::2])
        # Relative positions
        # self._dqs = jnp.expand_dims(self._qs[:,:,1] - self._qs[:,:,0], axis=-1)
        self._dqs = jnp.diff(self._qs, axis=-1)
        # Conjugate momenta
        self._ps = jnp.array(state[:,:,1::2])
        # Velocities
        self._vs = self._ps / jnp.expand_dims(self._m, 1)  # reshape m to fit shape of velocity
        # Accelerations
        self._accs = jnp.diff(self._vs, axis=1) / self._dt
        final_acc = jnp.expand_dims(self._accs[:,-1,:], axis=1) # duplicate final acceleration
        self._accs = jnp.concatenate((self._accs, final_acc), axis=1) # add copy of final acceleration to end of accs
        data = jnp.concatenate((self._qs, self._dqs,self._ps, self._accs), axis=-1)
        data = jax.lax.stop_gradient(data)

        self._data = data
    
    def _get_norm_stats(self):
        norm_stats = ml_collections.ConfigDict()
        norm_stats.position = ml_collections.ConfigDict({
            'mean': jnp.mean(self._qs),
            'std': jnp.std(self._qs),
        })
        norm_stats.velocity = ml_collections.ConfigDict({
            'mean': jnp.mean(self._vs),
            'std': jnp.std(self._vs),
        })
        norm_stats.acceleration = ml_collections.ConfigDict({
            'mean': jnp.mean(self._accs),
            'std': jnp.std(self._accs),
        })
        norm_stats.control = ml_collections.ConfigDict({
            'mean': jnp.mean(self._control),
            'std': jnp.std(self._control)
        })

        self._norm_stats = norm_stats
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([jnp.shape(self._qs)[-1]])
        self.n_edge = jnp.array([jnp.shape(self._dqs)[-1]])
        self.senders = jnp.arange(0, jnp.shape(self._qs)[-1]-1)
        self.receivers = jnp.arange(1, jnp.shape(self._qs)[-1])
    
    @jax.jit
    def get_graph(self, traj_idx, t) -> jraph.GraphsTuple:
        """ Need to make sure t > self._vel_history! """
        match self._mode:
            case 'acceleration':
                vs_history = []                
                [vs_history.append(self._vs[traj_idx, t-k]) for k in reversed(range(self._vel_history))]
                vs_history = jnp.asarray(vs_history).T

                control_history = []
                [control_history.append(self._control[traj_idx, t-k, 1::2]) for k in reversed(range(self._control_history))]
                control_history = jnp.asarray(control_history).T
                # Node features are current position, velocity history, current velocity
                nodes = jnp.column_stack((self._qs[traj_idx, t], vs_history, control_history))
                # Edge features are relative positions
                edges = self._dqs[traj_idx, t].reshape((-1,1))
                # Global features are time, q0, v0, a0
                # global_context = jnp.concatenate((jnp.array([t]), self._qs[traj_idx, 0], self._vs[traj_idx, 0], self._accs[traj_idx, 0])).reshape(-1,1)
            
                # Global features are None
                global_context = None
            case 'position':
                raise NotImplementedError
            
        graph =  jraph.GraphsTuple(
                    nodes=nodes,
                    edges=edges,
                    senders=self.senders,
                    receivers=self.receivers,
                    n_node=self.n_node,
                    n_edge=self.n_edge,
                    globals=global_context)
        
        graph = add_edges(graph, self._add_undirected_edges, self._add_self_loops)

        return graph
    
    @jax.jit
    def get_graph_batch(self, traj_idxs, t0s) -> Sequence[jraph.GraphsTuple]:
        def f(carry, idxs):
            return carry, self.get_graph(*idxs)
        
        _, graphs = jax.lax.scan(f, None, (traj_idxs, t0s))
        
        return graphs

    def tree_flatten(self):
        children = () # dynamic
        aux_data = (self._path, self._add_undirected_edges, self._add_self_loops, self._mode, self._vel_history, self._control_history, self._data, self._norm_stats, self._qs, self._dqs, self._ps, self._vs, self._accs, self._control, self._m, self._k, self._b, self._dt) # static
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(MSDGraphBuilder)
        obj._path                   = aux_data[0]
        obj._add_undirected_edges   = aux_data[1]
        obj._add_self_loops         = aux_data[2]
        obj._mode                   = aux_data[3]
        obj._vel_history            = aux_data[4]
        obj._control_history        = aux_data[5]
        obj._data                   = aux_data[6]
        obj._norm_stats             = aux_data[7]
        obj._qs                     = aux_data[8]
        obj._dqs                    = aux_data[9]
        obj._ps                     = aux_data[10]
        obj._vs                     = aux_data[11]
        obj._accs                   = aux_data[12]
        obj._control                = aux_data[13]
        obj._m                      = aux_data[14]
        obj._k                      = aux_data[15]
        obj._b                      = aux_data[16]
        obj._dt                     = aux_data[17]
        obj._setup_graph_params()
        return obj
    
@register_pytree_node_class
class VehiclePlatoonGraphBuilder(GraphBuilder):
    def __init__(
            self, path, seed, alpha, dt, m, noise_std, num_initial_states, senders, receivers
        ):
        self.seed = seed
        self.alpha = alpha
        self.dt = dt
        self.m = m
        self.noise_std = noise_std
        self.num_initial_states = num_initial_states
        self.senders = senders
        self.receivers = receivers
        super().__init__(path, add_undirected_edges=False, add_self_loops=True)
    
    def _load_data(self, path):
        A_11 = jnp.array([[1, self.dt], [0, 1]])
        A_ii = jnp.array([[1, self.dt], [self.alpha * self.dt / self.m, 1]]) # i = 2,3,4,5
        A_ij = jnp.array([[0, 0], [-self.alpha * self.dt / self.m, 0]]) # i = 2,3,4,5; j = i-1
        B_ii = jnp.array([[0, 0], [0, self.dt / self.m]]) # i = 1,2,3,4,5

        zeros = jnp.zeros((2,2))

        self.A = jnp.block([[A_11, zeros, zeros, zeros, zeros], 
                            [A_ij, A_ii, zeros, zeros, zeros],
                            [zeros, A_ij, A_ii, zeros, zeros],
                            [zeros, zeros, A_ij, A_ii, zeros],
                            [zeros, zeros, zeros, A_ij, A_ii]])
        
        self.B = jnp.block([[B_ii, zeros, zeros, zeros, zeros],
                            [zeros, B_ii, zeros, zeros, zeros],
                            [zeros, zeros, B_ii, zeros, zeros],
                            [zeros, zeros, zeros, B_ii, zeros],
                            [zeros, zeros, zeros, zeros, B_ii]])
        
        self.state_dim = len(self.A)
        self.num_nodes = self.state_dim // 2

        self.Q = jnp.eye(self.state_dim)
        self.R = 0.1 * jnp.eye(self.num_nodes)

        self.K, self.S, _ = dlqr(self.A, self.B, self.Q, 0.1 * jnp.eye(self.state_dim)) # TODO 
        
        # When x > 0, 1 > 2 > 3 > 4 > 5
        def random_initial_state(_, key):
            signs = jax.random.choice(key, jnp.array([-1, 1]), shape=(self.num_nodes+1,))
            signs = jnp.array([signs[0], signs[1], signs[0], signs[2], signs[0], signs[3], signs[0], signs[4], signs[0], signs[5]])            
            state0 = 0.5 * signs * (jnp.array([1, 0, 2, 0, 3, 0, 4, 0, 5, 0]) \
                                  + jax.random.normal(key, shape=(self.state_dim,)))
            
            return _, state0
        
        key = jax.random.key(self.seed)
        keys = jax.random.split(key, self.num_initial_states)
        _, initial_states = jax.lax.scan(random_initial_state, None, keys)
        
        self.initial_states = jnp.array(initial_states)

    def _get_norm_stats(self):
        num_timesteps = 100
        key = jax.random.key(self.seed)
        keys = jax.random.split(key, self.num_initial_states)
        _, controls = jax.vmap(self.get_optimal_state_trajectory, in_axes=(0,0,None))(keys, self.initial_states, num_timesteps)

        norm_stats = ml_collections.ConfigDict()
        norm_stats.control = ml_collections.ConfigDict({
            'mean': jnp.mean(controls),
            'std': jnp.std(controls)
        })

        self.norm_stats = norm_stats
    
    def dynamics_function(self, state, inputs):
        key, control = inputs
        D = self.noise_std * jax.random.normal(key,(self.num_nodes,))
        w = self.dt / self.m * jnp.array([0, D[0], 0, D[1], 0, D[2], 0, D[3], 0, D[4]])
        next_state = self.A @ state + self.B @ control + w
        return next_state, (next_state, control)
    
    def get_state_trajectory(self, key, state_0, controls, num_timesteps):
        # optimal control only depends on state
        keys = jax.random.split(key, num_timesteps)
        stateT, outputs = jax.lax.scan(self.dynamics_function, state_0, (keys, controls))
        states = jnp.array(outputs[0])
        # states = jnp.concatenate((state_0.reshape(1,-1), states))
        controls = jnp.array(outputs[1])

        return states, controls
    
    def optimal_dynamics_function(self, state, key):
        D = self.noise_std * jax.random.normal(key, (self.num_nodes,))
        w = self.dt / self.m * jnp.array([0, D[0], 0, D[1], 0, D[2], 0, D[3], 0, D[4]])
        control = -self.K @ state
        next_state = self.A @ state + self.B @ control + w
        return next_state, (next_state, control)
    
    def get_optimal_state_trajectory(self, key, state_0, num_timesteps):
        # optimal control only depends on state
        keys = jax.random.split(key, num_timesteps)
        state_T, outputs = jax.lax.scan(self.optimal_dynamics_function, state_0, keys) 
        states = jnp.array(outputs[0])
        # states = jnp.concatenate((state_0.reshape(1,-1), states))
        controls = jnp.array(outputs[1])

        return states, controls
    
    def _setup_graph_params(self):
        self.n_node = jnp.array([self.num_nodes])
        self.n_edge = self.n_node - 1
    
    def get_graph(self, state: jnp.array) -> jraph.GraphsTuple:
        """ Return a graph with node features being [x(k), u(k)] """
        nodes = state.reshape((self.num_nodes,-1))
        edges = None
        globals = None
        graph = jraph.GraphsTuple(nodes=nodes,
                                  edges=edges,
                                  globals=globals,
                                  senders=self.senders,
                                  receivers=self.receivers,
                                  n_node=self.n_node,
                                  n_edge=self.n_edge)
        return graph
    
    def get_graph_batch(self, states) -> Sequence[jraph.GraphsTuple]:
        def f(carry, idx):
            return carry, self.get_graph(idx)
        
        _, graphs = jax.lax.scan(f, None, states)
        
        return graphs
    
    def tree_flatten(self):
        children = () # dynamic
        aux_data = (self.seed, self._add_undirected_edges, self._add_self_loops, self.A, self.B, self.Q, self.R, self.K, self.S, self.initial_states, self.num_initial_states, self.state_dim, self.num_nodes, self.norm_stats) # static
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = object.__new__(VehiclePlatoonGraphBuilder)
        obj.seed                    = aux_data[0]
        obj._add_undirected_edges   = aux_data[1]
        obj._add_self_loops         = aux_data[2]
        obj.A                       = aux_data[3]
        obj.B                       = aux_data[4]
        obj.Q                       = aux_data[5]
        obj.R                       = aux_data[6]
        obj.K                       = aux_data[7]
        obj.S                       = aux_data[8]
        obj.initial_states          = aux_data[9]
        obj.num_initial_states      = aux_data[10]
        obj.state_dim               = aux_data[11]
        obj.num_nodes               = aux_data[12]
        obj.norm_states             = aux_data[13]
        obj._setup_graph_params()
        return obj
    
