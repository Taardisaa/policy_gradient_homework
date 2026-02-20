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
from loguru import logger

from core import MLPActorCritic, discount_cumsum

# Choose backend before importing pyplot.
# TkAgg works over SSH X11 without OpenGL/GLX; Agg is headless-only.
_live_render = '--render' in sys.argv and '--save-gif' not in sys.argv
matplotlib.use('TkAgg' if _live_render else 'Agg')
import matplotlib.pyplot as plt


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


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
          gamma=0.99, lam=0.97, vf_lr=1e-3, vf_iters=80,
          epochs=50, batch_size=5000, render=False, save_gif=False,
          checkpoint_dir=None):
    """
    Train the policy gradient.

    Args:
    - env_name: name of the gym environment to train on
    - hidden_sizes: list of hidden layer sizes for the policy network
    - lr: learning rate for policy optimization
    - gamma: discount factor for returns
    - lam: GAE lambda parameter
    - vf_lr: learning rate for value network
    - vf_iters: number of iterations to optimize value network per epoch
    - epochs: number of training epochs
    - batch_size: number of steps of interaction (state-action pairs) to collect per epoch
    - render: whether to render the environment during training
    - save_gif: whether to save per-epoch GIFs of the rendered environment (implies render=True)
    - checkpoint_dir: directory to save model checkpoints (one per epoch); if None, no checkpoints are saved
    """

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)


    device = 'cpu'  # just use CPU for now.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    ac = MLPActorCritic(env.observation_space, env.action_space,
                        hidden_sizes=hidden_sizes)
    ac.pi.to(device)
    ac.v.to(device) # also put value network on device.

    # ac.act() calls .numpy() internally which breaks on CUDA tensors.
    # We call the distribution directly and move back to CPU for env stepping.
    def get_action(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            # val = ac.v(obs_t).item()
        # ep_vals.append(val)  # save value estimates for GAE
        # batch_vals.append(val)  # also save for value function targets
            return ac.pi._distribution(obs_t).sample().cpu().numpy()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        obs, act, weights = obs.to(device), act.to(device), weights.to(device)
        _, logp = ac.pi(obs, act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(ac.pi.parameters(), lr=lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # run one episode to visualize the current policy.
    # Uses rgb_array rendering + matplotlib to avoid pygame/GLX/OpenGL,
    # which fails over SSH X11 forwarding on headless GPU servers.
    def render_episode(epoch, save_gif=False, gif_dir=f'results/extra_a/{env_name}/frames'):
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
                act = get_action(obs)
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
            logger.info(f'[render] saved {gif_path}')

        return ep_ret

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_vals = []         # for value function targets
        batch_adv = []          # for GAE advantages
        batch_rtgs = []          # for reward-to-go returns
        ep_vals = []           # value estimates for current episode (for GAE) 

        # reset episode-specific variables
        obs, info = env.reset()  # updated reset handling
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            # compute V(s_t) BEFORE stepping, so ep_vals[t] = V(s_t)
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                val = ac.v(obs_t).item()
            ep_vals.append(val)
            act = get_action(obs)
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
                # batch_weights += [ep_ret] * ep_len
                # batch_weights += list(reward_to_go(ep_rews))
                ep_rews_arr = np.array(ep_rews, dtype=np.float32)
                ep_vals_arr = np.array(ep_vals, dtype=np.float32)
                # bootstrap: 0 if truly done, else V(s_last) if truncated
                last_val = 0 if terminated else ac.v(
                    torch.as_tensor(obs, dtype=torch.float32).to(device)
                ).item()

                # TD residuals
                vals_with_boot = np.append(ep_vals_arr, last_val)
                deltas = ep_rews_arr + gamma * vals_with_boot[1:] - vals_with_boot[:-1]

                # GAE advantages
                ep_adv = discount_cumsum(deltas, gamma * lam)
                # Reward-to-go (for value target)
                ep_rtgs = discount_cumsum(ep_rews_arr, gamma)

                batch_adv += list(ep_adv)
                batch_rtgs += list(ep_rtgs)

                # reset episode-specific variables
                obs, info = env.reset()
                done, ep_rews, ep_vals = False, [], []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # normalize advantages before policy update
        adv_arr = np.array(batch_adv, dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        obs_t  = torch.as_tensor(np.array(batch_obs),  dtype=torch.float32).to(device)
        act_dtype = torch.int32 if isinstance(env.action_space, Discrete) else torch.float32
        acts_t = torch.as_tensor(np.array(batch_acts), dtype=act_dtype)
        adv_t  = torch.as_tensor(adv_arr,              dtype=torch.float32)
        rtgs_t = torch.as_tensor(np.array(batch_rtgs), dtype=torch.float32).to(device)

        # policy gradient step using GAE advantages as weights
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=obs_t, act=acts_t, weights=adv_t)
        batch_loss.backward()
        optimizer.step()

        # value function update: multiple gradient steps per epoch for stability
        for _ in range(vf_iters):
            vf_optimizer.zero_grad()
            vf_loss = ((ac.v(obs_t) - rtgs_t) ** 2).mean()
            vf_loss.backward()
            vf_optimizer.step()

        return batch_loss.item(), batch_rets, batch_lens

    # training loop
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    history = {'epoch': [], 'return': []}
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        mean_ret = np.mean(batch_rets)
        history['epoch'].append(i)
        history['return'].append(mean_ret)
        logger.info('epoch: {:3d} | loss: {:.3f} | return: {:.3f} | ep_len: {:.3f}'.format(
                i, batch_loss, mean_ret, np.mean(batch_lens)))
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{i:03d}.pt')
            torch.save(ac.pi.state_dict(), ckpt_path)
        if render:
            ep_ret = render_episode(i, save_gif=save_gif)
            logger.info(f'[render] episode return: {ep_ret:.1f}')

    return history


def save_chart(history, out_path, env_name, run=None, max_return=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['epoch'], history['return'], color='steelblue', linewidth=2)
    if max_return is not None:
        ax.axhline(max_return, color='gray', linestyle='--', linewidth=1,
                   label=f'Max Return ({max_return})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Episode Return')
    title = f'Policy Gradient (Part 3) — {env_name}'
    if run is not None:
        title += f' (run {run})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f'Chart saved to {out_path}')


def main_argparse():
    """
    Parse command-line arguments and execute the main program with parsed parameters.
    Command-line arguments:
        --env_name, --env (str): Name of the OpenAI Gym environment to use. 
                                 Default: 'CartPole-v1'
        --render (bool): If set, render the environment during execution.
                         Default: False
        --save-gif (bool): If set, save per-epoch GIFs instead of live rendering.
                          Useful for headless servers. Default: False
        --lr (float): Learning rate for the policy gradient optimizer.
                      Default: 1e-2
        --epochs (int): Number of training epochs.
                        Default: 50
        --num-runs (int): Number of independent runs to execute.
                          Default: 1
    Returns:
        The return value of main() function with the parsed arguments.
    Note:
        Uses the simplest formulation of policy gradient for training.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-gif', action='store_true',
                        help='save per-epoch GIFs instead of live render (use on headless servers)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-runs', type=int, default=1)
    args = parser.parse_args()
    logger.info('Using simplest formulation of policy gradient.')

    return main(
        env_name=args.env_name, 
        render=args.render,
        save_gif=args.save_gif,
        lr=args.lr,
        epochs=args.epochs,
        num_runs=args.num_runs)


def main(
    env_name='CartPole-v1', 
    render=False, 
    save_gif=False, 
    lr=1e-2, 
    epochs=50, 
    num_runs=1
):
    """
    Execute training and evaluation pipeline for reinforcement learning policy gradient algorithm.

    Manages directory structure, runs multiple training iterations, saves checkpoints,
    and generates performance visualizations.

    Args:
        env_name (str, optional): Name of the OpenAI Gym environment. Defaults to 'CartPole-v1'.
        render (bool, optional): Whether to render environment during training. Defaults to False.
        save_gif (bool, optional): Whether to save training episodes as GIF. Defaults to False.
        lr (float, optional): Learning rate for the policy gradient optimizer. Defaults to 1e-2.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        num_runs (int, optional): Number of independent training runs to execute. Defaults to 1.

    Returns:
        None

    Side Effects:
        - Creates directory structure at 'results/extra_a/{env_name}/'
        - Saves training checkpoints to checkpoint directory
        - Saves performance charts as PNG files
        - Prints run information to console when num_runs > 1
        - Skips existing run results (detected by output PNG file existence)
    """
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/extra_a', exist_ok=True)
    os.makedirs(f'results/extra_a/{env_name}', exist_ok=True)
    for run in range(num_runs):
        if num_runs > 1:
            logger.info(f'--- Run {run} ---')
        out_path = f'results/extra_a/{env_name}/run{run}.png' if num_runs > 1 else f'results/extra_a/{env_name}.png'
        if Path(out_path).exists():
            continue
        ckpt_dir = f'results/extra_a/{env_name}/checkpoints' if num_runs == 1 else f'results/extra_a/{env_name}/checkpoints/run{run}'
        history = train(env_name=env_name, render=render or save_gif,
                        save_gif=save_gif, lr=lr, epochs=epochs,
                        checkpoint_dir=ckpt_dir)
        save_chart(history, out_path, env_name, run=run if num_runs > 1 else None)


if __name__ == '__main__':
    main_argparse()


