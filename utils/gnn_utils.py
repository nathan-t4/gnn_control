import os
import flax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import Dict, Any
from clu import metrics

@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
  """Adds a prefix to the keys of a dict, returning a new dict."""
  return {f'{prefix}_{key}': val for key, val in result.items()}

def plot_evaluation_curves(
        ts, pred_data, exp_data, aux_data, prefix, plot_dir, prediction='control', show=False
    ):
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if prediction == 'acceleration':
        m = jnp.round(aux_data[0], 3)
        k = jnp.round(aux_data[1], 3)
        b = jnp.round(aux_data[2], 3)

        q0 = jnp.round(exp_data[0,0], 3)
        a0 = jnp.round(exp_data[1,0], 3)

        N = len(m)
        """ Error Plot """
        # fig, (ax1, ax2) = plt.subplots(2,1)
        # fig.suptitle(f'{prefix}: Eval Error \n q0 = {q0}, a0 = {a0}')

        # ax1.set_title('Position')
        # ax1.plot(ts, exp_data[0] - pred_data[0], label=[f'Mass {i}' for i in range(2)])
        # ax1.set_xlabel('Time [$s$]')
        # ax1.set_ylabel('Position error [$m$]')
        # ax1.legend()

        # ax2.set_title('Acceleration')
        # ax2.plot(ts, exp_data[1] - pred_data[1], label=[f'Mass {i}' for i in range(2)])
        # ax2.set_xlabel('Time [$s$]')
        # ax2.set_ylabel(r'Acceleration error [$\mu m/s^2$]')
        # ax2.legend()

        # plt.tight_layout()
        # fig.savefig(os.path.join(plot_dir, f'{prefix}_error.png'))
        # plt.show() if show else plt.close()

        for i in range(N):
            fig, (ax1, ax2) = plt.subplots(2,1)
            title = f"{prefix}: Mass {i} \n $m_{i}$ = " + "{:.2f},".format(m[i]) + f" $k_{i}$ = " + "{:.2f},".format(k[i]) + f" $b_{i}$ = " + "{:.2f}".format(b[i])
            fig.suptitle(title)
            ax1.set_title(f'Position')
            ax1.plot(ts, pred_data[0,:,i], label='predicted')
            ax1.plot(ts, exp_data[0,:,i], label='expected')
            ax1.set_xlabel('Time [$s$]')
            ax1.set_ylabel('Position [$m$]')
            ax1.legend()

            ax2.set_title(f'Acceleration')
            ax2.plot(ts, pred_data[1,:,i], label='predicted')
            ax2.plot(ts, exp_data[1,:,i], label='expected')
            ax2.set_xlabel('Time [$s$]')
            ax2.set_ylabel(r'Acceleration [$\mu m/s^2$]')
            ax2.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{prefix}_mass{i}.png'))
            if show: plt.show()
            plt.close()
    
    elif prediction == 'control':
        predicted_state = (pred_data[0]).squeeze()
        predicted_control = pred_data[1]
        expected_state = exp_data[0]
        expected_control = exp_data[1]

        # plot state & control for both 1. predicted 2. expected
        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.suptitle('Predicted using Graph Network')
        ax1.set_title(f'Position')
        ax1.plot(ts, predicted_state[:,::2], label=['1','2','3','4','5'])
        ax1.set_xlabel(r'$t$ [$s$]')
        ax1.set_ylabel(r'$x$ [$m$]')
        ax1.legend()

        ax2.set_title(f'Control')
        ax2.plot(ts, predicted_control, label=['1','2','3','4','5'])
        ax2.set_xlabel(r'$t$ [$s$]')
        ax2.set_ylabel(r'$u$ [$N$]')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_predicted.png'))
        if show: plt.show()
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.suptitle('Optimal State Feedback')
        ax1.set_title(f'Position')
        ax1.plot(ts, expected_state[:,::2], label=['1','2','3','4','5'])
        ax1.set_xlabel(r'$t$ [$s$]')
        ax1.set_ylabel(r'$x$ [$m$]')
        ax1.legend()

        ax2.set_title(f'Control')
        ax2.plot(ts, expected_control[:,1::2], label=['1','2','3','4','5'])
        ax2.set_xlabel(r'$t$ [$s$]')
        ax2.set_ylabel(r'$u$ [$N$]')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_expected.png'))
        if show: plt.show()
        plt.close()
        
    plt.close()