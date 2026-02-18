import sys
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import os
import matplotlib
from pathlib import Path

# Choose backend before importing pyplot.
# TkAgg works over SSH X11 without OpenGL/GLX; Agg is headless-only.
_live_render = '--render' in sys.argv and '--save-gif' not in sys.argv
matplotlib.use('TkAgg' if _live_render else 'Agg')
import matplotlib.pyplot as plt


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """
    Implementation of MLP.
    """
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False, save_gif=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # run one episode to visualize the current policy.
    # Uses rgb_array rendering + matplotlib to avoid pygame/GLX/OpenGL,
    # which fails over SSH X11 forwarding on headless GPU servers.
    def render_episode(epoch, save_gif=False, gif_dir='results/vanilla_pg/frames'):
        # Tell SDL to render offscreen — no window, no GLX needed.
        os.environ['SDL_VIDEODRIVER'] = 'offscreen'
        render_env = gym.make(env_name, render_mode="rgb_array")
        obs, info = render_env.reset()
        done = False
        ep_ret = 0
        frames = []

        if not save_gif:
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis('off')
            img_display = ax.imshow(render_env.render())
            ax.set_title(f'Epoch {epoch}')
            plt.tight_layout()
            plt.pause(0.001)

        while not done:
            frame = render_env.render()
            frames.append(frame)
            if not save_gif:
                img_display.set_data(frame)
                plt.pause(0.05)  # ~20 fps
            with torch.no_grad():
                act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = render_env.step(act)
            done = terminated or truncated
            ep_ret += rew

        render_env.close()

        if not save_gif:
            plt.ioff()
            plt.close(fig)

        if save_gif and frames:
            import imageio
            os.makedirs(gif_dir, exist_ok=True)
            gif_path = os.path.join(gif_dir, f'epoch_{epoch:03d}.gif')
            imageio.mimsave(gif_path, frames, fps=30)
            print(f'  [render] saved {gif_path}')

        return ep_ret

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, info = env.reset()  # updated reset handling
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, info = env.reset()
                done, ep_rews = False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss.item(), batch_rets, batch_lens

    # training loop
    history = {'epoch': [], 'return': []}
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        mean_ret = np.mean(batch_rets)
        history['epoch'].append(i)
        history['return'].append(mean_ret)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, mean_ret, np.mean(batch_lens)))
        if render:
            ep_ret = render_episode(i, save_gif=save_gif)
            print(f'  [render] episode return: {ep_ret:.1f}')

    return history


def save_chart(history, out_path, env_name, run=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['epoch'], history['return'], color='steelblue', linewidth=2)
    ax.axhline(500, color='gray', linestyle='--', linewidth=1, label='Max Return (500)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Episode Return')
    title = f'Simple Policy Gradient — {env_name}'
    if run is not None:
        title += f' (run {run})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Chart saved to {out_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-gif', action='store_true',
                        help='save per-epoch GIFs instead of live render (use on headless servers)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num-runs', type=int, default=1)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/vanilla_pg', exist_ok=True)
    for run in range(args.num_runs):
        if args.num_runs > 1:
            print(f'\n--- Run {run} ---')
        out_path = f'results/vanilla_pg/return_run{run}.png' if args.num_runs > 1 else 'results/vanilla_pg/return.png'
        if Path(out_path).exists():
            continue
        history = train(env_name=args.env_name, render=args.render or args.save_gif,
                        save_gif=args.save_gif, lr=args.lr)
        save_chart(history, out_path, args.env_name, run=run if args.num_runs > 1 else None)
