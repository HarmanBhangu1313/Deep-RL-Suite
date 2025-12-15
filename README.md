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
## installations
git clone <https://github.com/HarmanBhangu1313/Deep-RL-Suite>
cd deep-rl-suite
pip install -r requirements.txt
