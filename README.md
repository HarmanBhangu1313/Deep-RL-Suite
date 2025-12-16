# Deep Reinforcement Learning Suite  
### DQN • A3C • PPO • SAC (PyTorch)

This repository is a **modular Deep Reinforcement Learning framework** built entirely from scratch.  
It implements four major RL families:

- **Value-Based** → DQN, Double DQN, Dueling DQN  
- **On-Policy Actor–Critic** → A3C, PPO  
- **Off-Policy Actor–Critic** → SAC  
- **Shared Core Utilities** → wrappers, replay buffer, GAE, logging, etc.

The suite is designed to be **clean, extensible, and experiment-ready**, with consistent structure across algorithms and environments.

---

##  Repository Structure
deep-rl-suite/
│
├── README.md               # this file
├── requirements.txt        # project dependencies
│
├── core/
│   ├── wrappers.py         # Env preprocessing (frame stack, grayscale, resize)
│   ├── replay_buffer.py    # Uniform & n-step replay buffer
│   ├── gae.py              # Generalized Advantage Estimation
│   ├── utils.py            # seeding, tensor helpers, discounting
│   ├── logger.py           # TensorBoard & PNG export
│
├── dqn/
│   ├── agent.py            # DQN agent with target network
│   ├── network.py          # MLP/CNN Q-networks
│   ├── train_dqn.py        # DQN training script
│   └── README.md
│
├── a3c/
│   ├── global_network.py   # CNN Actor–Critic shared network
│   ├── worker.py           # A3C worker logic
│   ├── train_a3c.py        # Synchronous A3C/A2C training loop
│   └── README.md
│
├── ppo/
│   ├── policy.py           # CNN Actor–Critic (mean, std, value)
│   ├── agent.py            # PPO agent (GAE + clipping)
│   ├── train_ppo.py        # PPO training loop
│   └── README.md
│
└── sac/
├-- sac.py                  # full sac implementation
└── README.md

---

##  Key Features

### **✔ Modular Architecture**
Every algorithm follows a common structure:
- `agent.py`
- `network/policy.py`
- `train_*.py`

Algorithms can be swapped without changing the env preprocessing or logging.

---

### **✔ Unified Preprocessing Pipeline**
Located in `core/wrappers.py`:
- Gray-scaling  
- Resize  
- Frame stacking  
- Reward clipping  
- Normalization  
- Max-and-skip  

Fully compatible with Atari-like or custom pixel environments.

---



## Algorithm Implemented

DQN Family
	•	Experience Replay
	•	Target Networks
	•	Double DQN
	•	Dueling Networks
	•	CNN/MLP support

A3C
	•	Shared global network
	•	Multi-env synchronous workers
	•	Entropy regularization

PPO
	•	Gaussian policy
	•	Generalized Advantage Estimation (GAE)
	•	Clipped surrogate loss
	•	Multi-epoch minibatch training

SAC
	•	Automatic entropy tuning
	•	Twin Q-networks
	•	Soft target updates
	•	Off-policy replay
    
---

## Environment Validation

To demonstrate robustness and generalization, the algorithms are evaluated on multiple environments with different dynamics, reward structures, and action spaces:

Discrete Control (Atari-style)
	•	Kung-Fu Master
	•	Pacman

Continuous Control (Box2D)
	•	LunarLander-v2

These environments test:
	•	sparse vs dense rewards
	•	discrete vs continuous action spaces
	•	stability under noisy gradients

---

##  Mathematical Intuition (Deep Reinforcement Learning Suite)

This project implements multiple **Deep Reinforcement Learning (Deep RL)** algorithms to study how agents learn optimal behavior through interaction with an environment.

At a high level, all algorithms aim to **maximize cumulative reward** by learning from trial and error.

---

###  Reinforcement Learning Setup
In Reinforcement Learning, an agent repeatedly:
- Observes a state from the environment
- Takes an action
- Receives a reward
- Transitions to a new state

The objective is not to maximize immediate reward, but to learn a **policy** that maximizes **long-term cumulative reward**.

---

###  DQN – Learning Action Values
Deep Q-Networks (DQN) learn a function that estimates:
- How good it is to take a specific action in a given state

Instead of learning a policy directly, DQN learns **action values** and selects the action with the highest expected return.

Key ideas:
- Neural networks approximate value functions
- A target network stabilizes learning
- Experience replay reduces correlation between samples

DQN works well for **discrete action spaces**.

---

###  A3C – Learning Policy and Value Together
Asynchronous Advantage Actor-Critic (A3C) separates learning into two parts:
- **Actor**: learns which action to take
- **Critic**: evaluates how good the current state is

Multiple agents interact with the environment in parallel, which:
- Improves exploration
- Stabilizes training
- Reduces sample correlation

A3C directly learns a **policy**, rather than relying on value lookup.

---

###  PPO – Stable Policy Optimization
Proximal Policy Optimization (PPO) improves policy-gradient methods by **restricting how much the policy can change in a single update**.

Instead of allowing large, unstable updates:
- PPO limits updates to a safe range
- This prevents performance collapse
- Training becomes more stable and reliable

PPO is widely used because it balances:
- Simplicity
- Stability
- Performance

---

###  SAC – Entropy-Regularized Learning
Soft Actor-Critic (SAC) introduces **entropy maximization** into the learning objective.

This encourages the agent to:
- Explore more
- Avoid premature convergence
- Learn robust behaviors

Key ideas:
- Learns both value and policy networks
- Encourages stochastic (diverse) actions
- Works well in **continuous action spaces**

SAC trades determinism for **robust exploration**.

---

###  Why Multiple Algorithms Are Implemented
Each algorithm represents a different philosophy:

- DQN → Value-based learning  
- A3C → On-policy actor-critic  
- PPO → Stable policy gradients  
- SAC → Exploration-driven learning  

Implementing all of them highlights how modern Deep RL evolved to address:
- Instability
- Sample inefficiency
- Exploration challenges

---

###  Multi-Environment Validation
The algorithms are tested on multiple environments (e.g., Atari and control tasks) to demonstrate:
- Generalization across dynamics
- Discrete vs continuous action handling
- Robustness of learned policies

---

###  Key Limitation
Deep RL methods:
- Are sample-inefficient
- Require careful tuning
- Can be unstable without proper design choices

This project prioritizes **understanding learning behavior**, not production optimization.

---

## installations
git clone <https://github.com/HarmanBhangu1313/Deep-RL-Suite>
cd deep-rl-suite
pip install -r requirements.txt
