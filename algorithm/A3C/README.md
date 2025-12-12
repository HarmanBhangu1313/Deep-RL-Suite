# A3C — Asynchronous Advantage Actor–Critic  
Part of the **Deep Reinforcement Learning Algorithm Suite (DQN → PPO → A3C)**

This folder contains a modular implementation of **A3C-style Actor–Critic**, using:

- A shared **CNN Actor–Critic network** (`global_network.py`)  
- A worker module (`worker.py`) implementing policy/value updates  
- A vectorized, synchronous training loop (`train_a3c.py`) that follows the same update pattern as A3C/A2C  
- Support for pixel-based environments (stacked frames, grayscale, normalization)  
- Integrated logging via `RLLogger` (TensorBoard + PNG graphs)

**Note:** This version uses *synchronous multi-env rollouts* (A2C-style). A full multiprocessing A3C version can be enabled easily if needed.

---

##  Folder Contents

```
a3c/
│── global_network.py     # Shared CNN Actor–Critic model (logits + value)
│── worker.py             # Worker class (policy/value update step)
│── train_a3c.py          # Main training loop (vectorized environments)
│── README.md             # This file
```

The A3C implementation depends on reusable utilities located in:

```
core/
│── wrappers.py           # Frame stacking, resize, grayscale, clipping
│── utils.py              # set_seed, tensor helpers
│── logger.py             # TensorBoard + PNG export
```

---

##  Algorithm Summary

A3C (Asynchronous Advantage Actor–Critic) is a policy gradient method where multiple workers explore in parallel, each computing:

- A stochastic **policy** π(a | s)  
- A **value estimate** V(s)  
- **Advantages** A(s,a) using bootstrapped value estimates  
- **Policy loss**:  
  $begin:math:display$
  L\_\{\\text\{actor\}\} \= \-\\log \\pi\(a\|s\) \\cdot A
  $end:math:display$
- **Value loss**:  
  $begin:math:display$
  L\_\{\\text\{critic\}\} \= \(V \- \\text\{target\}\)\^2
  $end:math:display$
- Optional **entropy bonus** encourages exploration.

This implementation uses **Categorical(logits)** for action sampling and computes advantages with bootstrapped returns (GAE is optional).

---

##  Network Architecture (`global_network.py`)

The network includes:

- 3× CNN layers for pixel observations  
- Adaptive pooling (`7×7`) to avoid hardcoded flatten sizes  
- Fully-connected layer (128 units)  
- **Actor head** → outputs logits `[B, action_size]`  
- **Critic head** → outputs state-value `[B, 1]`

Key features:
- Orthogonal initialization (stable for actor–critic)
- Channel-first input `(C, H, W)`  
- Float normalization (`state/255.0`)

---

##  Worker Logic (`worker.py`)

Each worker performs:

1. **Forward pass** to get logits + values  
2. **Sample action** using `Categorical(logits)`  
3. **Compute targets** using:
   $begin:math:display$
   R \+ \\gamma V\(s\_\{\\text\{next\}\}\)
   $end:math:display$
4. **Compute advantages**:
   $begin:math:display$
   A \= \\text\{target\} \- V\(s\)
   $end:math:display$
5. **Policy / value / entropy losses**  
6. **Backprop & gradient clipping**

Returned losses (for logging):

```
{
  "actor_loss": ...,
  "critic_loss": ...,
  "entropy": ...,
  "total_loss": ...,
  "avg_value": ...
}
```

---

##  Training (`train_a3c.py`)

Uses **vectorized environments (EnvBatch or gym.vector)** to simulate multiple environments per step.

### Training flow:
- Create N environments (e.g., N = 16)
- Collect `t_max` steps per environment  
- Flatten rollouts → batch size = `num_envs * t_max`  
- Call `worker.update_from_batch(batch)`  
- Log metrics with `RLLogger`  
- Periodically evaluate policy on a single env

### Rollout collection
At each step:
```
actions = worker.act(batch_states)
next_states, rewards, dones, _ = env_batch.step(actions)
```

### Logging
Integrated with `RLLogger`:

- `actor_loss`
- `critic_loss`
- `entropy`
- `avg_episode_return`
- `eval_mean_return`

Exports PNG graphs to:

```
experiments/graphs/
```

---

## How to Run

**Option 1 — Direct run of A3C training script**
```bash
python a3c/train_a3c.py \
    --num_envs 16 \
    --t_max 5 \
    --total_updates 20000 \
    --reward_scale 0.01 \
    --run_name a3c_LunarLander
```

**Option 2 — Through main orchestrator (`main.py`)**
```bash
python main.py --algo a3c --env LunarLander-v2
```

(Ensure `main.py` dispatches to `train_a3c` correctly.)

---

##  Evaluation

During training:
```bash
[Update 2000] Eval mean return: 132.4
```

Manual evaluation of a trained model:
```python
returns = evaluate(worker, env, n_episodes=10)
print(np.mean(returns))
```

---

## TODO / Extensions

- Add **GAE-Lambda** (Generalized Advantage Estimation) for smoother advantage estimates.  
- Connect workers to a shared global network (true multiprocessing A3C).  
- Add CNN-LSTM version for partially observable environments.  
- Add video recording for trained agent (`gymnasium.wrappers.RecordVideo`).

---

###  Summary

This folder implements a clean, modular A3C system that integrates with your Deep RL suite.  
It includes:

✔ CNN global network  
✔ Worker logic  
✔ Vectorized rollout trainer  
✔ Logging, evaluation, checkpointing  
✔ Easy extension to multiprocessing

---
