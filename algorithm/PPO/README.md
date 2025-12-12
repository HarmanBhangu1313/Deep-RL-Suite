# Proximal Policy Optimization (PPO)
Part of the **Deep Reinforcement Learning Algorithm Suite** (DQN → A3C → PPO → SAC)

This folder contains a full PPO implementation supporting:
- Continuous action spaces (e.g., CarRacing-v2)
- CNN Actor–Critic network
- Gaussian stochastic policy
- Generalized Advantage Estimation (GAE)
- Multi-epoch minibatch PPO updates
- Rollout collection with bootstrap values
- Logging, evaluation, and checkpointing

---

##  Folder Structure

ppo/
│── policy.py          # CNN Actor–Critic (mean, std, value)
│── agent.py           # PPOAgent: rollout buffer, GAE, PPO update
│── train_ppo.py       # Main training loop
│── README.md          # This file

---

## Shared modules:
core/
│── wrappers.py        # preprocess (resize, frame-stack, normalize)
│── logger.py          # TensorBoard + PNG export
│── utils.py           # seed, device helpers

---
---

## 🧠 Algorithm Overview

**PPO** is an on-policy actor–critic method that maintains stability using:
- **Clipped objective**  
- **GAE advantages**  
- **Entropy regularization**  
- **Multiple epochs per rollout**

### Clipped Surrogate Objective
\[
L = \min \left( r_t A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right)
\]

### GAE Advantages
\[
A_t = \sum_k (\gamma \lambda)^k \delta_{t+k}
\]

---

##  Network (policy.py)

The Actor–Critic network contains:
- 3-layer CNN encoder  
- Fully connected latent layer  
- **Actor head** → Gaussian `mean` + learned `log_std`  
- **Critic head** → scalar state value  
- Orthogonal initialization  
- Supports CHW image input  

---

##  PPOAgent (agent.py)

Handles:
- Rollout storage (`states, actions, logprobs, rewards, dones, values`)
- GAE advantage computation
- PPO update with:
  - Ratio and clip objective  
  - Value loss  
  - Entropy bonus  
  - Gradient clipping  
- Old policy sync after updates
- Continuous action sampling with `Independent(Normal)`

---

##  Training Loop (train_ppo.py)

The training loop:
1. Collects `buffer_size` timesteps  
2. Computes bootstrap value for last state  
3. Runs `agent.ppo_update()`  
4. Logs progress and optionally renders  
5. Saves checkpoints if enabled  

Uses Gymnasium’s step API:
obs, reward, terminated, truncated, info
done = terminated or truncated

---

##  Usage

### Basic training:
```bash
python ppo/train_ppo.py --env CarRacing-v2 --total_timesteps 200000
python ppo/train_ppo.py --render
python ppo/train_ppo.py --buffer_size 4096 --mini_batch_size 128