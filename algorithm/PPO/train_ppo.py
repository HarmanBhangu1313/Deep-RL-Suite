# ppo/train_ppo.py
import os
import time
import argparse
from collections import deque

import numpy as np
import torch

# adjust imports to your project structure
# from core.wrappers import make_env
# from core.logger import RLLogger
# from core.utils import set_seed, get_device
# from ppo.agent import PPOAgent, Memory  # your agent implementation
# from ppo.policy import ActorCritic     # ActorCritic network class

# Fallbacks if you don't have get_device helper:
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

def parse_args():
    p = argparse.ArgumentParser("PPO training")
    p.add_argument("--env", type=str, default="CarRacing-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--buffer_size", type=int, default=2048, help="timesteps per PPO update (rollout length)")
    p.add_argument("--mini_batch_size", type=int, default=64)
    p.add_argument("--ppo_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--render", action="store_true")
    p.add_argument("--logdir", type=str, default="experiments/logs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_every", type=int, default=0, help="save checkpoint every N updates (0=never)")
    return p.parse_args()


def train(args):
    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or f"ppo_{args.env}_{int(time.time())}"
    log_dir = os.path.join(args.logdir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # create env (use wrapper if you have one)
    import gymnasium as gym
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    # optionally use wrappers for frames / normalization
    # env = make_env(args.env, seed=args.seed, frame_stack=4, width=84, height=84, normalize=False)

    # build agent
    obs_space = env.observation_space
    action_space = env.action_space
    # instantiate your PPOAgent (ensure its constructor accepts obs_space/action_space and hyperparams)
    agent = PPOAgent(obs_space, action_space)  # if your agent requires config, pass it

    # logger
    # logger = RLLogger(log_dir=log_dir, config={"algo": "ppo", "env": args.env})
    # if you don't have RLLogger, you can just print

    # training bookkeeping
    total_timesteps = args.total_timesteps
    buffer_size = args.buffer_size
    timestep = 0
    episodes = 0
    recent_returns = deque(maxlen=100)
    ep_reward = 0.0
    ep_len = 0

    # initialize environment
    obs, info = env.reset()
    state = preprocess_observation(obs)  # keep same preprocessing convention used in agent.select_action

    # main loop
    update_idx = 0
    last_log_time = time.time()
    while timestep < total_timesteps:
        # Collect rollout of length buffer_size
        for _ in range(buffer_size):
            # select action from policy (old policy is used inside select_action in your agent)
            # agent.select_action returns action_clipped (numpy), logprob (tensor cpu), value (tensor cpu)
            action, logprob, value = agent.select_action(state)

            # step environment: gymnasium returns (obs, reward, terminated, truncated, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # store transition in agent.memory
            agent.store_transition(state, action, logprob, float(reward), done, value)

            if args.render:
                env.render()

            # advance
            state = preprocess_observation(next_obs)
            ep_reward += float(reward)
            ep_len += 1
            timestep += 1

            if done:
                recent_returns.append(ep_reward)
                episodes += 1
                # reset episode counters
                obs, info = env.reset()
                state = preprocess_observation(obs)
                ep_reward = 0.0
                ep_len = 0

            if timestep >= total_timesteps:
                break

        # Bootstrap last value if last stored step not terminal
        last_val = torch.tensor(0.0)
        if len(agent.memory.dones) > 0 and not agent.memory.dones[-1]:
            # last state in memory
            last_state = agent.memory.states[-1]
            if isinstance(last_state, np.ndarray):
                last_state_t = torch.from_numpy(last_state).float().unsqueeze(0).to(DEVICE)
            else:
                last_state_t = last_state.unsqueeze(0).to(DEVICE) if last_state.dim() == 3 else last_state.to(DEVICE)
            with torch.no_grad():
                _, _, last_val = agent.actor_critic(last_state_t)
            last_val = last_val.squeeze(0).cpu()

        # perform PPO update (agent.ppo_update should use agent.memory)
        agent.ppo_update()

        update_idx += 1

        # optional logging
        if time.time() - last_log_time > 10:
            avg_ret = float(np.mean(recent_returns)) if len(recent_returns) > 0 else float('nan')
            print(f"[t={timestep}/{total_timesteps}] updates={update_idx} episodes={episodes} avg_return_100={avg_ret:.2f}")
            last_log_time = time.time()

        # optional checkpointing
        if args.save_every > 0 and (update_idx % args.save_every == 0):
            ckpt_base = os.path.join(log_dir, f"ppo_update_{update_idx}")
            torch.save(agent.actor_critic.state_dict(), ckpt_base + "_policy.pth")
            try:
                torch.save(agent.optimizer.state_dict(), ckpt_base + "_opt.pth")
            except Exception:
                pass

    # final save
    torch.save(agent.actor_critic.state_dict(), os.path.join(log_dir, "ppo_final_policy.pth"))
    try:
        torch.save(agent.optimizer.state_dict(), os.path.join(log_dir, "ppo_final_opt.pth"))
    except Exception:
        pass

    env.close()
    print("Training complete.")
    return agent


def preprocess_observation(obs):
    """
    Minimal default preprocessing: convert to CHW float tensor in [0,1] if image,
    otherwise return as-is (e.g., low-dim vector) — adapt to your project.
    """
    if isinstance(obs, np.ndarray):
        if obs.ndim == 3:  # HWC -> CHW, normalize
            arr = np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0
            return arr
        else:
            return obs.astype(np.float32)
    return obs


if __name__ == "__main__":
    args = parse_args()
    train(args)