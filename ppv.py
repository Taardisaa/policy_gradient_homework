"""
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
"""

import sys
import numpy as np
import torch
from torch.optim import Adam
from loguru import logger
import gymnasium as gym
import time
import os
import matplotlib
from pathlib import Path

import core

# Choose backend before importing pyplot.
# TkAgg works over SSH X11 without OpenGL/GLX; Agg is headless-only.
_live_render = '--render' in sys.argv and '--save-gif' not in sys.argv
matplotlib.use('TkAgg' if _live_render else 'Agg')
import matplotlib.pyplot as plt


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick

        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        # Added a small epsilon 1e-8 to prevent division by zero.
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def save_chart(history, out_path, env_name, run=None, max_return=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['epoch'], history['return'], color='steelblue', linewidth=2)
    if max_return is not None:
        ax.axhline(max_return, color='gray', linestyle='--', linewidth=1,
                   label=f'Max Return ({max_return})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Episode Return')
    title = f'PPO — {env_name}'
    if run is not None:
        title += f' (run {run})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f'Chart saved to {out_path}')


def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10, render=False, save_gif=False, checkpoint_dir=None):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        render (bool): Whether to render one episode per epoch for visualization.

        save_gif (bool): Whether to save per-epoch GIFs instead of live rendering.

        checkpoint_dir (str): Directory to save model checkpoints. If None, no
            checkpoints are saved.

    """

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    env_name = env.spec.id
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.info(f'Number of parameters: pi: {var_counts[0]}, v: {var_counts[1]}')

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Simple stats accumulator replacing EpochLogger.store / log_tabular / dump_tabular
    stats = {}

    def store(**kwargs):
        for k, v in kwargs.items():
            stats.setdefault(k, []).append(v)

    def dump_stats(epoch, elapsed):
        def mean(k): return np.mean(stats[k]) if k in stats else float('nan')
        def minmax(k): return (np.min(stats[k]), np.max(stats[k])) if k in stats else (float('nan'), float('nan'))
        ep_ret_min, ep_ret_max = minmax('EpRet')
        vval_min, vval_max = minmax('VVals')
        logger.info(
            f"Epoch {epoch} | "
            f"EpRet {mean('EpRet'):.2f} [{ep_ret_min:.2f}, {ep_ret_max:.2f}] | "
            f"EpLen {mean('EpLen'):.1f} | "
            f"VVals {mean('VVals'):.3f} [{vval_min:.3f}, {vval_max:.3f}] | "
            f"TotalEnvInteracts {(epoch+1)*steps_per_epoch} | "
            f"LossPi {mean('LossPi'):.4f} | LossV {mean('LossV'):.4f} | "
            f"DeltaLossPi {mean('DeltaLossPi'):.4f} | DeltaLossV {mean('DeltaLossV'):.4f} | "
            f"Entropy {mean('Entropy'):.3f} | KL {mean('KL'):.4f} | "
            f"ClipFrac {mean('ClipFrac'):.3f} | StopIter {mean('StopIter'):.1f} | "
            f"Time {elapsed:.1f}s"
        )
        mean_ep_ret = mean('EpRet')
        stats.clear()
        return mean_ep_ret

    def render_episode(epoch, save_gif=False, gif_dir=None):
        os.environ['SDL_VIDEODRIVER'] = 'offscreen'
        render_env = gym.make(env_name, render_mode='rgb_array')
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
                act = ac.act(torch.as_tensor(obs, dtype=torch.float32))
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

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)  # pi(a|s) / pi_old(a|s), which is $r_t(\theta)$ in the PPO paper.
        # clip
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        # Approximate KL Divergence.
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                logger.info(f'Early stopping at step {i} due to reaching max kl.')
                break
            loss_pi.backward()
            pi_optimizer.step()

        store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        store(LossPi=pi_l_old, LossV=v_l_old,
              KL=kl, Entropy=ent, ClipFrac=cf,
              DeltaLossPi=(loss_pi.item() - pi_l_old),
              DeltaLossV=(loss_v.item() - v_l_old))

    # Auto-resume from latest checkpoint if one exists.
    start_epoch = 0
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_files = sorted(Path(checkpoint_dir).glob('epoch_*.pt'))
        if ckpt_files:
            latest = ckpt_files[-1]
            ckpt = torch.load(latest, map_location='cpu')
            ac.pi.load_state_dict(ckpt['pi'])
            ac.v.load_state_dict(ckpt['v'])
            pi_optimizer.load_state_dict(ckpt['pi_opt'])
            vf_optimizer.load_state_dict(ckpt['vf_opt'])
            start_epoch = ckpt['epoch'] + 1
            logger.info(f'Resumed from {latest} (continuing from epoch {start_epoch})')

    # Prepare for interaction with environment
    start_time = time.time()
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0

    history = {'epoch': [], 'return': []}

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(start_epoch, epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    logger.warning(f'Trajectory cut off by epoch at {ep_len} steps.')
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    store(EpRet=ep_ret, EpLen=ep_len)
                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

        # Save model checkpoint
        if checkpoint_dir and ((epoch % save_freq == 0) or (epoch == epochs - 1)):
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pt')
            torch.save({
                'pi': ac.pi.state_dict(),
                'v': ac.v.state_dict(),
                'pi_opt': pi_optimizer.state_dict(),
                'vf_opt': vf_optimizer.state_dict(),
                'epoch': epoch,
            }, ckpt_path)
            logger.info(f'Checkpoint saved to {ckpt_path}')

        # Perform PPO update!
        update()

        # Log epoch stats and capture mean return for history
        mean_ep_ret = dump_stats(epoch, time.time() - start_time)
        history['epoch'].append(epoch)
        history['return'].append(mean_ep_ret)

        if render or save_gif:
            gif_dir = os.path.join(checkpoint_dir or f'results/ppo/{env_name}', 'frames') if save_gif else None
            ep_ret_render = render_episode(epoch, save_gif=save_gif, gif_dir=gif_dir)
            logger.info(f'[render] episode return: {ep_ret_render:.1f}')

    return history


def main(env_name='HalfCheetah-v2', hid=64, l=2, gamma=0.99, seed=0,
         steps=4000, epochs=50, render=False, save_gif=False, num_runs=1):
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/ppo', exist_ok=True)
    os.makedirs(f'results/ppo/{env_name}', exist_ok=True)
    for run in range(num_runs):
        if num_runs > 1:
            logger.info(f'--- Run {run} ---')
        out_path = (f'results/ppo/{env_name}/run{run}.png' if num_runs > 1
                    else f'results/ppo/{env_name}.png')
        ckpt_dir = (f'results/ppo/{env_name}/checkpoints/run{run}' if num_runs > 1
                    else f'results/ppo/{env_name}/checkpoints')
        history = ppo(
            lambda: gym.make(env_name),
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[hid]*l),
            gamma=gamma,
            seed=seed,
            steps_per_epoch=steps,
            epochs=epochs,
            render=render or save_gif,
            save_gif=save_gif,
            checkpoint_dir=ckpt_dir,
        )
        save_chart(history, out_path, env_name, run=run if num_runs > 1 else None)


def main_argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-gif', action='store_true',
                        help='save per-epoch GIFs instead of live render (use on headless servers)')
    parser.add_argument('--num-runs', type=int, default=1)
    args = parser.parse_args()
    return main(
        env_name=args.env_name,
        hid=args.hid,
        l=args.l,
        gamma=args.gamma,
        seed=args.seed,
        steps=args.steps,
        epochs=args.epochs,
        render=args.render,
        save_gif=args.save_gif,
        num_runs=args.num_runs,
    )


if __name__ == '__main__':
    main_argparse()
