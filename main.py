import argparse
import json
import os
import time
import importlib

import gymnasium as gym
from core.wrappers import make_env  


def load_config(path):
    """Load JSON/YAML config file if provided."""
    if path.endswith(".json"):
        return json.load(open(path, "r"))
    elif path.endswith(".yaml") or path.endswith(".yml"):
        import yaml
        return yaml.safe_load(open(path, "r"))
    return {}


def get_algorithm_module(algo_name):
    """
    Dynamically import algorithm train script.
    
    """
    algo = algo_name.lower()
    try:
        return importlib.import_module(f"algorithms.{algo}.train_{algo}")
    except Exception as e:
        raise ImportError(f"[ERROR] Algorithm '{algo}' not found: {e}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, required=True,
                        help="Algorithm: dqn, double_dqn, dueling_dqn, a3c, ppo")

    parser.add_argument("--env", type=str, required=True,
                        help="Gym environment ID: LunarLander-v2, Pacman-v0, KungFuMaster-v0")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (.json or .yaml)")

    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for logging folder")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load custom config if provided
    config = load_config(args.config) if args.config else {}

    # Generate run name automatically if not provided
    if args.run_name is None:
        args.run_name = f"{args.algo}_{args.env}_{int(time.time())}"

    # Make logs directory
    log_dir = os.path.join("experiments/logs", args.run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save config for reproducibility
    config_to_save = {"algo": args.algo, "env": args.env, "seed": args.seed}
    config_to_save.update(config)
    json.dump(config_to_save, open(os.path.join(log_dir, "config.json"), "w"), indent=2)

    # Environment
    env = make_env(args.env, seed=args.seed, **config.get("env_wrappers", {}))

    # Dynamically load training module
    algo_module = get_algorithm_module(args.algo)

    print(f"\n Starting training:")
    print(f"   Algorithm    : {args.algo}")
    print(f"   Environment  : {args.env}")
    print(f"   Run Name     : {args.run_name}")
    print(f"   Logs saved to: {log_dir}\n")

    # Call algorithm's train function
    algo_module.train(env=env, log_dir=log_dir, config=config)


if __name__ == "__main__":
    main()