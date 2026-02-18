"""
compare.py — run Vanilla PG and Reward-to-Go PG each N times and plot
their learning curves (mean ± std) on the same axes.
"""

import os
import sys
import numpy as np

# Set backend before any module imports pyplot.
# Both vanilla_pg and reward_to_go also call matplotlib.use('Agg') at import
# time (when --render is absent), so setting it here first is safe.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import train functions with distinct names.
import vanilla_pg as vpg_mod
import reward_to_go as rtg_mod


def run_experiment(train_fn, name, n_runs, epochs, batch_size, lr):
    all_returns = []
    for run in range(n_runs):
        print(f'\n[{name}] run {run + 1}/{n_runs}')
        history = train_fn(epochs=epochs, batch_size=batch_size, lr=lr,
                           render=False, save_gif=False)
        all_returns.append(history['return'])
    return np.array(all_returns)   # shape: (n_runs, epochs)


def plot_comparison(results, epochs, batch_size, out_path):
    # x-axis: approximate cumulative environment timesteps
    # each epoch collects ~batch_size steps before the gradient update
    timesteps = (np.arange(epochs) + 1) * batch_size

    fig, ax = plt.subplots(figsize=(10, 5))
    styles = [
        ('Vanilla PG',       results['vpg'], 'steelblue'),
        ('Reward-to-Go PG',  results['rtg'], 'coral'),
    ]
    for label, returns, color in styles:
        mean = returns.mean(axis=0)
        std  = returns.std(axis=0)
        ax.plot(timesteps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(timesteps, mean - std, mean + std,
                        alpha=0.2, color=color)

    ax.axhline(500, color='gray', linestyle='--', linewidth=1,
               label='Max Return (500)')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Return')
    ax.set_title('Vanilla PG vs Reward-to-Go PG — CartPole-v1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'\nComparison chart saved to {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs',     type=int,   default=5)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=5000)
    parser.add_argument('--lr',         type=float, default=1e-2)
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    vpg_path = 'results/vpg_returns.npy'
    rtg_path = 'results/rtg_returns.npy'

    # Run vanilla PG (skip if cached)
    if os.path.exists(vpg_path):
        print(f'Loading cached VPG results from {vpg_path}')
        vpg_returns = np.load(vpg_path)
    else:
        vpg_returns = run_experiment(vpg_mod.train, 'Vanilla PG',
                                     args.n_runs, args.epochs,
                                     args.batch_size, args.lr)
        np.save(vpg_path, vpg_returns)
        print(f'Saved {vpg_path}')

    # Run reward-to-go PG (skip if cached)
    if os.path.exists(rtg_path):
        print(f'Loading cached RTG results from {rtg_path}')
        rtg_returns = np.load(rtg_path)
    else:
        rtg_returns = run_experiment(rtg_mod.train, 'Reward-to-Go PG',
                                     args.n_runs, args.epochs,
                                     args.batch_size, args.lr)
        np.save(rtg_path, rtg_returns)
        print(f'Saved {rtg_path}')

    plot_comparison(
        {'vpg': vpg_returns, 'rtg': rtg_returns},
        epochs=vpg_returns.shape[1],
        batch_size=args.batch_size,
        out_path='results/comparison.png',
    )
