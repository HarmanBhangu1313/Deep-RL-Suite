# dqn/train_dqn.py
import os
import time
import argparse
from collections import deque

import numpy as np
import torch

from core.wrappers import make_env
from core.logger import RLLogger
from core.utils import set_seed
from dqn.agent import DQNAgent   


def parse_args():
    parser = argparse.ArgumentParser("DQN Training")
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max_t", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--solved_score", type=float, default=200.0)  # typical for LunarLander
    parser.add_argument("--logdir", type=str, default="experiments/logs")
    return parser.parse_args()


def train(agent, env, logger, args, config):
    """
    Main training loop for DQN.

    agent: instance of DQNAgent
    env: gym env (already wrapped)
    logger: RLLogger
    args: parsed CLI args
    config: hyperparameters dict (optional)
    """
    n_episodes = args.episodes
    max_t = args.max_t

    eps_start = config.get("eps_start", 1.0)
    eps_end = config.get("eps_end", 0.01)
    eps_decay = config.get("eps_decay", 0.995)
    epsilon = eps_start

    scores_deque = deque(maxlen=100)
    best_avg = -np.inf

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()  # gymnasium: (obs, info)
        score = 0.0

        for t in range(max_t):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores_deque.append(score)
        avg_score = float(np.mean(scores_deque))

        # Logging
        logger.log_scalar("episode_reward", float(score), step=episode)
        logger.log_scalar("avg_reward_100", float(avg_score), step=episode)
        logger.log_scalar("epsilon", float(epsilon), step=episode)

        # print / progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d}\tScore: {score:7.2f}\tAvg100: {avg_score:7.2f}\tEps: {epsilon:.3f}")

        # save best model (by average)
        if avg_score > best_avg:
            best_avg = avg_score
            save_path = os.path.join(logger.log_dir, "best_model.pth")
            agent.save(save_path)

        # periodic checkpoint
        if episode % args.save_every == 0:
            ckpt_path = os.path.join(logger.log_dir, f"checkpoint_ep{episode}.pth")
            agent.save(ckpt_path)

        # decay epsilon
        epsilon = max(eps_end, eps_decay * epsilon)

        # optional solved condition
        if avg_score >= args.solved_score and episode >= 100:
            print(f"\nEnvironment solved in {episode - 100} episodes! Avg100 = {avg_score:.2f}")
            agent.save(os.path.join(logger.log_dir, "solved_model.pth"))
            break

    # export plots and close logger
    logger.export_graphs(out_dir="experiments/graphs")
    logger.close()


def main():
    args = parse_args()

    # create run name
    run_name = args.run_name or f"dqn_{args.env}_{int(time.time())}"
    log_dir = os.path.join(args.logdir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # basic config / hyperparams (tweak as needed)
    config = {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": int(1e5),
        "tau": 1e-3,
        "update_every": 4,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.995,
        "n_step": 1,
    }

    # seeds
    set_seed(args.seed)

    # create env (use wrappers)
    env = make_env(args.env, seed=args.seed, frame_stack=1, normalize=True)

    # instantiate agent
    # state_shape used by agent: for MLP use int state dim, for CNN we'll pass tuple
    # Here we assume low-dim (vector) env; adjust if you're using pixel input.
    state_shape = env.observation_space.shape
    if len(state_shape) == 1:
        state_dim = int(state_shape[0])
        agent = DQNAgent(state_size=(state_dim,), action_size=env.action_space.n, config=config)
    else:
        # pixel input (H, W, C) or (H,W) stacked -> agent/network must support CNN
        agent = DQNAgent(state_size=state_shape, action_size=env.action_space.n, config=config)

    # logger
    logger = RLLogger(log_dir=log_dir, config={"algo": "dqn", "env": args.env, **config})

    # start training
    train(agent, env, logger, args, config)


if __name__ == "__main__":
    main()