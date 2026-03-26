# 📑 Technical Documentation: Intelligent Systems Algorithms

This document provides a deep dive into the theoretical background and implementation details of the algorithms used in the **Robotics** and **Reinforcement Learning** sections of this project.

---

## 🤖 Section A: Robotics & Intelligent Agents

### 1. Grid-Based Navigation
The `navigate_robot` function implements a discrete state-space movement on a 2D grid.
- **State Space**: $S \in \{(r, c) \mid 0 \le r < 10, 0 \le c < 10\}$.
- **Action Space**: $A \in \{\text{UP, DOWN, LEFT, RIGHT}\}$.
- **Clamping Logic**: To prevent the agent from leaving the environment boundaries, we use:
  $$r_{new} = \max(0, \min(9, r + \Delta r))$$
  $$c_{new} = \max(0, \min(9, c + \Delta c))$$

### 2. Weighted Sensor Fusion
The `sensor_fusion` function simulates how multiple noisy sensors are combined into a single steering command.
- **Weights**: 
  - $w_{left} = -1.0$ (Steer left)
  - $w_{center} = 0.0$ (Neutral)
  - $w_{right} = 1.0$ (Steer right)
- **Decision Rule**:
  $$\text{weighted\_sum} = \sum_{i \in \{L, C, R\}} (s_i \cdot w_i)$$
  The robot steers based on a **dead-zone threshold** ($\tau = 0.1$) to ignore minor sensor fluctuations:
  - If $\text{sum} > \tau \implies \text{RIGHT}$
  - If $\text{sum} < -\tau \implies \text{LEFT}$
  - Otherwise $\implies \text{STRAIGHT}$

---

## 💡 Section B: Reinforcement Learning Fundamentals

### 1. The Bellman Equation (Q-Learning)
Q-Learning is an **off-policy** temporal difference (TD) control algorithm. The goal is to learn the action-value function $Q(s, a)$.
- **Update Rule**:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
- **Parameters**:
  - $\alpha$ (Learning Rate): Determines how much new information overrides old.
  - $\gamma$ (Discount Factor): Determines the importance of future rewards.

### 2. Softmax Action Selection
To balance exploration and exploitation, we use the Gibbs/Boltzmann distribution:
$$P(a | s) = \frac{\exp(Q(s, a) / \tau)}{\sum_{j} \exp(Q(s, j) / \tau)}$$
- **Temperature ($\tau$)**: 
  - $\tau \to \infty$: Actions are chosen nearly uniformly (Pure Exploration).
  - $\tau \to 0$: The probability of the best action approaches 1.0 (Pure Exploitation).

---

## 🏆 Section C: Capstone Integrated Agent

### 1. Epsilon-Greedy Strategy
The agent in `section_c.py` uses an $\epsilon$-greedy policy with decay:
- With probability $\epsilon$, choose a random action.
- With probability $1-\epsilon$, choose $a = \arg\max_a Q(s, a)$.
- **Decay**: $\epsilon_{t+1} = \max(\epsilon_{min}, \epsilon_t \cdot \epsilon_{decay})$.

### 2. Environment Dynamics
The 1D world simulates a Markov Decision Process (MDP) where:
- **Rewards**: 
  - $R = +10$ if $s' = \text{Goal}$
  - $R = -1$ otherwise (Time penalty).
- Training for 50 episodes allows the agent to build a Q-table that reflects the shortest path to the goal.

---

## 🛠 Tools and Utilities
- **NumPy**: Used for matrix operations in the Policy Network and Q-table management.
- **He Initialization**: Used in `PolicyNetwork` to initialize weights:
  $$\sigma = \sqrt{\frac{2}{\text{fan\_in}}}$$
  This prevents internal covariate shift and keeps signal variance stable through ReLU layers.
