"""
visualize.py — load a saved policy checkpoint and replay it.

Examples:
  # Watch a single epoch (live window via X11):
  python visualize.py --env Ant-v5 --checkpoint results/part_3/Ant-v5/checkpoints/run0/epoch_049.pt

  # Save a GIF (headless server):
  python visualize.py --env Ant-v5 --checkpoint results/part_3/Ant-v5/checkpoints/run0/epoch_049.pt --save-gif

  # Sweep all checkpoints in a directory and save a GIF per epoch:
  python visualize.py --env Ant-v5 --checkpoint-dir results/part_3/Ant-v5/checkpoints/run0 --save-gif
"""

import sys
import os

# Must be set before any gym/mujoco import.
# EGL = headless OpenGL on NVIDIA GPU (no display needed).
# Fall back to osmesa (software) if EGL is unavailable.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'offscreen')

import argparse
import torch
import numpy as np
import gymnasium as gym
import matplotlib
from pathlib import Path

# Choose backend before importing pyplot.
# TkAgg works over SSH X11 without OpenGL/GLX; Agg is headless-only.
_live_render = '--save-gif' not in sys.argv and '--plot' not in sys.argv
matplotlib.use('TkAgg' if _live_render else 'Agg')
import matplotlib.pyplot as plt

from core import MLPActorCritic


def load_policy(env, checkpoint_path, hidden_sizes):
    ac = MLPActorCritic(env.observation_space, env.action_space,
                        hidden_sizes=hidden_sizes)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # Support both full checkpoint dicts (ppv.py) and bare pi state dicts (extra_a.py).
    if isinstance(ckpt, dict) and 'pi' in ckpt:
        ac.pi.load_state_dict(ckpt['pi'])
        ac.v.load_state_dict(ckpt['v'])
    else:
        ac.pi.load_state_dict(ckpt)
    ac.pi.eval()
    return ac


def get_action(ac, obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        return ac.pi._distribution(obs_t).sample().numpy()


def run_episode(ac, env_name, save_gif=False, gif_path=None, n_episodes=1):
    """Run n_episodes and return a list of episode returns."""
    render_env = gym.make(env_name, render_mode='rgb_array')

    returns = []
    for ep in range(n_episodes):
        obs, info = render_env.reset()
        done = False
        ep_ret = 0
        frames = []

        if not save_gif:
            plt.ion()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.axis('off')
            img_display = ax.imshow(render_env.render())
            plt.tight_layout()
            plt.pause(0.001)

        while not done:
            frame = render_env.render()
            frames.append(frame)
            if not save_gif:
                img_display.set_data(frame)
                plt.pause(0.033)   # ~30 fps
            act = get_action(ac, obs)
            obs, rew, terminated, truncated, info = render_env.step(act)
            done = terminated or truncated
            ep_ret += rew

        returns.append(ep_ret)
        print(f'  episode {ep}: return = {ep_ret:.2f}')

        if not save_gif:
            plt.ioff()
            plt.close(fig)

        if save_gif and gif_path and frames:
            import imageio
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            imageio.mimsave(gif_path, frames, fps=30)
            print(f'  saved {gif_path}')

    render_env.close()
    return returns


def eval_episode(ac, env_name, n_episodes=1):
    """Run n_episodes without rendering and return a list of episode returns."""
    eval_env = gym.make(env_name)
    returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0
        while not done:
            act = get_action(ac, obs)
            obs, rew, terminated, truncated, _ = eval_env.step(act)
            done = terminated or truncated
            ep_ret += rew
        returns.append(ep_ret)
    eval_env.close()
    return returns


def save_sweep_chart(all_returns, out_path, env_name):
    epochs = [e for e, _ in all_returns]
    means = [r for _, r in all_returns]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, means, color='steelblue', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Episode Return')
    ax.set_title(f'PPO Eval — {env_name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Chart saved to {out_path}')


def sweep_checkpoints(checkpoint_dir, env_name, hidden_sizes, save_gif, out_dir,
                      n_episodes=1):
    """Load every epoch_*.pt in a directory and run one episode each."""
    ckpts = sorted(Path(checkpoint_dir).glob('epoch_*.pt'))
    if not ckpts:
        print(f'No checkpoints found in {checkpoint_dir}')
        return []

    env = gym.make(env_name)
    all_returns = []
    for ckpt in ckpts:
        epoch = int(ckpt.stem.split('_')[1])
        print(f'\nEpoch {epoch:03d}  ({ckpt})')
        ac = load_policy(env, ckpt, hidden_sizes)
        if save_gif:
            gif_path = os.path.join(out_dir, f'epoch_{epoch:03d}.gif')
            rets = run_episode(ac, env_name, save_gif=True, gif_path=gif_path,
                               n_episodes=n_episodes)
        else:
            rets = eval_episode(ac, env_name, n_episodes=n_episodes)
        mean_ret = np.mean(rets)
        all_returns.append((epoch, mean_ret))
        print(f'  mean return = {mean_ret:.2f}')
    env.close()

    print('\n--- sweep summary ---')
    for epoch, r in all_returns:
        print(f'  epoch {epoch:03d}: mean return = {r:.2f}')

    return all_returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True,
                        help='Gymnasium env name, e.g. Ant-v5 or CartPole-v1')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single .pt checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory of epoch_*.pt files — sweeps all of them')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[32],
                        help='Must match the hidden_sizes used during training (default: 32)')
    parser.add_argument('--n-episodes', type=int, default=1,
                        help='Number of episodes to run per checkpoint')
    parser.add_argument('--save-gif', action='store_true',
                        help='Save GIF(s) instead of live display')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory for GIFs (default: next to checkpoint)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot mean eval return per epoch and save as PNG (requires --checkpoint-dir)')
    parser.add_argument('--out-chart', type=str, default=None,
                        help='Path for the eval chart PNG (default: <checkpoint-dir>/eval_return.png)')
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_dir is None:
        parser.error('Provide --checkpoint or --checkpoint-dir')

    hidden_sizes = tuple(args.hidden_sizes)

    if args.checkpoint_dir:
        out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'gifs')
        all_returns = sweep_checkpoints(args.checkpoint_dir, args.env, hidden_sizes,
                                        save_gif=args.save_gif, out_dir=out_dir,
                                        n_episodes=args.n_episodes)
        if args.plot and all_returns:
            out_chart = args.out_chart or os.path.join(args.checkpoint_dir, 'eval_return.png')
            save_sweep_chart(all_returns, out_chart, args.env)
    else:
        env = gym.make(args.env)
        ac = load_policy(env, args.checkpoint, hidden_sizes)
        env.close()
        stem = Path(args.checkpoint).stem
        out_dir = args.out_dir or str(Path(args.checkpoint).parent / 'gifs')
        gif_path = os.path.join(out_dir, f'{stem}.gif') if args.save_gif else None
        print(f'\nRunning {args.n_episodes} episode(s) with {args.checkpoint}')
        run_episode(ac, args.env, save_gif=args.save_gif,
                    gif_path=gif_path, n_episodes=args.n_episodes)
