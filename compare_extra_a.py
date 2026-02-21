"""
compare_extra_a.py — run Part 3 (no baseline) and Extra A (GAE baseline) each
N times on one or more environments and plot their learning curves (mean ± std).

Results are cached as .npy files so re-running is fast.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import part_3 as p3_mod
import extra_a as ea_mod

MAX_RETURNS = {
    'CartPole-v1':        500,
    'InvertedPendulum-v5': 1000,
}


def run_experiment(train_fn, name, env_name, n_runs, epochs, lr, cache_path):
    if os.path.exists(cache_path):
        print(f'Loading cached {name} results from {cache_path}')
        return np.load(cache_path)

    all_returns = []
    for run in range(n_runs):
        print(f'\n[{name} | {env_name}] run {run + 1}/{n_runs}')
        history = train_fn(env_name=env_name, epochs=epochs, lr=lr,
                           render=False, save_gif=False)
        all_returns.append(history['return'])

    arr = np.array(all_returns)   # shape: (n_runs, epochs)
    np.save(cache_path, arr)
    print(f'Saved {cache_path}')
    return arr


def plot_comparison(p3_returns, ea_returns, env_name, epochs, batch_size, out_path):
    timesteps = (np.arange(epochs) + 1) * batch_size

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, returns, color in [
        ('Part 3 (no baseline)',  p3_returns, 'steelblue'),
        ('Extra A (GAE baseline)', ea_returns, 'coral'),
    ]:
        mean = returns.mean(axis=0)
        std  = returns.std(axis=0)
        ax.plot(timesteps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)

    max_ret = MAX_RETURNS.get(env_name)
    if max_ret is not None:
        ax.axhline(max_ret, color='gray', linestyle='--', linewidth=1,
                   label=f'Max Return ({max_ret})')

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Return')
    ax.set_title(f'Part 3 vs Extra A (GAE) — {env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Comparison chart saved to {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+',
                        default=['CartPole-v1', 'InvertedPendulum-v5'])
    parser.add_argument('--n-runs',     type=int,   default=3)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=5000)
    parser.add_argument('--lr',         type=float, default=1e-2)
    args = parser.parse_args()

    os.makedirs('results/extra_a', exist_ok=True)

    for env_name in args.envs:
        cache_dir = f'results/extra_a/{env_name}'
        os.makedirs(cache_dir, exist_ok=True)

        p3_cache = f'{cache_dir}/p3_returns.npy'
        ea_cache = f'{cache_dir}/ea_returns.npy'

        p3_returns = run_experiment(
            p3_mod.train, 'Part3', env_name,
            args.n_runs, args.epochs, args.lr, p3_cache)

        ea_returns = run_experiment(
            ea_mod.train, 'ExtraA', env_name,
            args.n_runs, args.epochs, args.lr, ea_cache)

        out_path = f'{cache_dir}/comparison.png'
        plot_comparison(p3_returns, ea_returns, env_name,
                        epochs=p3_returns.shape[1],
                        batch_size=args.batch_size,
                        out_path=out_path)
