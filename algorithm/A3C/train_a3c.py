# a3c/train_a3c.py
import time
import argparse
from collections import deque

import numpy as np
import torch
import tqdm

from a3c.worker import A3CWorker   
from core.logger import RLLogger
from core.utils import set_seed

# --- helper to unpack EnvBatch.step robustly ---
def step_and_unpack(env_batch, actions):
    """
    Call env_batch.step(actions) and handle different return signatures:
    - (next_states, rewards, dones, infos)
    - (next_states, rewards, terminateds, truncateds, infos)
    Returns: next_states, rewards, dones (boolean array), infos
    """
    out = env_batch.step(actions)
    # If env_batch.step returns a 4-tuple
    if len(out) == 4:
        next_states, rewards, dones, infos = out
    elif len(out) == 5:
        next_states, rewards, terminateds, truncateds, infos = out
        dones = np.logical_or(terminateds, truncateds)
    else:
        raise RuntimeError(f"Unsupported EnvBatch.step() return length: {len(out)}")
    return next_states, np.asarray(rewards), np.asarray(dones, dtype=np.bool_), infos


def evaluate(worker, single_env, n_episodes=10, max_steps=1000):
    """Run greedy policy evaluation (no exploration). Returns list of episode returns."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = single_env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            # worker.act expects numpy state shape (C,H,W) or batch; ensure single-state
            action = worker.act(obs)  # returns array; for single env will be shape (1,)
            if isinstance(action, np.ndarray):
                action = int(action[0])
            obs, reward, terminated, truncated, info = single_env.step(action)
            done = bool(terminated or truncated)
            total += float(reward)
            steps += 1
        returns.append(total)
    return returns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--t_max", type=int, default=5, help="steps per update (rollout length)")
    p.add_argument("--total_updates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--logdir", type=str, default="experiments/logs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--reward_scale", type=float, default=0.01)
    p.add_argument("--in_channels", type=int, default=4)
    p.add_argument("--action_size", type=int, default=None)  # you can set this or infer later
    return p.parse_args()


def train(env_batch, single_env, worker, logger, args):
    """
    env_batch: vectorized env with reset() -> batch_states and step(actions) -> next_states,rewards,dones,info
    single_env: a single Gym env used for evaluation
    worker: instance of A3CWorker with method act() and update_from_batch()
    logger: RLLogger
    args: parsed CLI args
    """
    num_envs = args.num_envs
    t_max = args.t_max
    total_updates = args.total_updates
    reward_scale = args.reward_scale

    # initialize
    batch_states = env_batch.reset()  # expect shape (num_envs, ...)
    # if env returns (obs, info)
    if isinstance(batch_states, tuple):
        batch_states = batch_states[0]

    # storage for running episode returns for display
    episode_returns = deque(maxlen=100)
    per_env_returns = np.zeros(num_envs, dtype=float)

    global_step = 0
    start_time = time.time()

    for update_idx in tqdm.trange(total_updates, desc="A3C updates"):
        # collect rollout of t_max steps
        rollout_states = []
        rollout_actions = []
        rollout_rewards = []
        rollout_next_states = []
        rollout_dones = []

        for step in range(t_max):
            # worker.act should accept batch of states and return actions array shape (num_envs,)
            actions = worker.act(batch_states)  # numpy array (num_envs,) or list
            actions = np.asarray(actions, dtype=np.int64)

            next_states, rewards, dones, infos = step_and_unpack(env_batch, actions)

            # optional reward scaling
            if reward_scale != 1.0:
                rewards = rewards * reward_scale

            # accumulate per-env episode returns (for logging)
            per_env_returns += rewards
            # if any env done, record the return and zero that env's accumulator
            for i, d in enumerate(dones):
                if d:
                    episode_returns.append(per_env_returns[i])
                    per_env_returns[i] = 0.0

            rollout_states.append(batch_states)
            rollout_actions.append(actions)
            rollout_rewards.append(rewards)
            rollout_next_states.append(next_states)
            rollout_dones.append(dones)

            batch_states = next_states
            global_step += num_envs

        # flatten rollout to single batch
        # shape after stacking: (t_max, num_envs, ...)
        # we want flattened arrays of shape (t_max * num_envs, ...)
        states_flat = np.concatenate([s if isinstance(s, np.ndarray) else np.asarray(s) for s in rollout_states], axis=0)
        actions_flat = np.concatenate(rollout_actions, axis=0)
        rewards_flat = np.concatenate(rollout_rewards, axis=0)
        next_states_flat = np.concatenate([s if isinstance(s, np.ndarray) else np.asarray(s) for s in rollout_next_states], axis=0)
        dones_flat = np.concatenate(rollout_dones, axis=0).astype(np.float32)

        # Create batch dict expected by worker.update_from_batch
        batch = {
            "states": states_flat,
            "actions": actions_flat,
            "rewards": rewards_flat,
            "next_states": next_states_flat,
            "dones": dones_flat,
        }

        # perform update on worker (returns losses dict for logging)
        loss_info = worker.update_from_batch(batch)

        # logging scalars
        logger.log_scalar("actor_loss", loss_info.get("actor_loss", 0.0), step=update_idx)
        logger.log_scalar("critic_loss", loss_info.get("critic_loss", 0.0), step=update_idx)
        logger.log_scalar("entropy", loss_info.get("entropy", 0.0), step=update_idx)
        logger.log_scalar("avg_value", loss_info.get("avg_value", 0.0), step=update_idx)
        # also log average episode return (if any finished)
        if len(episode_returns) > 0:
            logger.log_scalar("avg_episode_return", float(np.mean(episode_returns)), step=update_idx)

        # periodic evaluation
        if update_idx % args.eval_every == 0:
            eval_returns = evaluate(worker, single_env, n_episodes=5)
            mean_eval = float(np.mean(eval_returns))
            print(f"\n[Update {update_idx}] Eval mean return: {mean_eval:.2f}")
            logger.log_scalar("eval_mean_return", mean_eval, step=update_idx)
            # checkpoint model
            ckpt = f"a3c_ckpt_update_{update_idx}.pth"
            worker.save(ckpt)

    # export graphs at the end
    logger.export_graphs(out_dir="experiments/graphs")
    logger.close()
    elapsed = time.time() - start_time
    print(f"Training finished. Time elapsed: {elapsed/60:.2f} minutes.")


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"a3c_{int(time.time())}"
    log_dir = f"{args.logdir}/{run_name}"

    logger = RLLogger(log_dir=log_dir, config={"algo": "a3c", "num_envs": args.num_envs})
    # NOTE: You must provide an EnvBatch class that matches the call pattern below.
    # env_batch = EnvBatch(number_environments)
    # single_env = gym.make(env_id)  # used for evaluation
    # For demo purposes assume user will construct env_batch and single_env outside and call train()

    # Create worker
    worker = A3CWorker(in_channels=args.in_channels, action_size=args.action_size or 6, config={"gamma":0.99})
    # The user should create env_batch and single_env and then call train()

    print("Prepared worker and logger. Please call train(env_batch, single_env, worker, logger, args).")

if __name__ == "__main__":
    main()