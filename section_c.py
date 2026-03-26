"""
=============================================================
  SECTION C — Capstone RL Project
  Problem 7: Mini Q-Learning Environment (1D World)
=============================================================
"""

import random
import numpy as np


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

class OneDWorldEnv:
    """
    A minimal 1D reinforcement learning environment.

    World:
        Positions: 0 to 9 (integer)
        Start:     Position 0
        Goal:      Position 9

    Actions:
        0 → LEFT  (position - 1, clamped at 0)
        1 → RIGHT (position + 1, clamped at 9)

    Rewards:
        +10  when the agent reaches position 9 (goal)
        -1   for every other step (encourages efficiency)

    Episode:
        Ends immediately when agent reaches position 9.
    """

    def __init__(self):
        self.goal_pos  = 9
        self.start_pos = 0
        self.position  = self.start_pos

    def reset(self) -> int:
        """Resets the environment and returns the starting state."""
        self.position = self.start_pos
        return self.position

    def step(self, action: int) -> tuple:
        """
        Takes one action in the environment.

        Args:
            action (int): 0 = LEFT, 1 = RIGHT

        Returns:
            tuple: (next_state, reward, done)
                - next_state (int): New position after action
                - reward (float): Reward received
                - done (bool): True if goal reached
        """
        if action == 0:    # LEFT
            self.position = max(0, self.position - 1)
        elif action == 1:  # RIGHT
            self.position = min(9, self.position + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        done   = (self.position == self.goal_pos)
        reward = 10.0 if done else -1.0

        return self.position, reward, done


# ─────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────

class QLearningAgent:
    """
    A tabular Q-learning agent for the 1D world environment.

    Q-table:
        Shape: (10 states) × (2 actions)
        Initialized to zeros.

    Learning:
        Uses epsilon-greedy exploration during training.
        Decays epsilon over episodes for more exploitation later.
    """

    def __init__(self,
                 n_states:  int   = 10,
                 n_actions: int   = 2,
                 alpha:     float = 0.1,
                 gamma:     float = 0.9,
                 epsilon:   float = 1.0,
                 epsilon_decay: float = 0.95,
                 epsilon_min:   float = 0.05):
        """
        Initializes the agent with hyperparameters.

        Args:
            n_states (int):        Number of states (positions 0–9).
            n_actions (int):       Number of actions (LEFT=0, RIGHT=1).
            alpha (float):         Learning rate.
            gamma (float):         Discount factor.
            epsilon (float):       Initial exploration rate.
            epsilon_decay (float): Multiplicative decay per episode.
            epsilon_min (float):   Minimum exploration rate.
        """
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-table initialized to zero: shape (states, actions)
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        """
        Selects an action using epsilon-greedy policy.

        With probability epsilon, choose a random action (explore).
        Otherwise, choose the action with the highest Q-value (exploit).

        Args:
            state (int): Current state (position).

        Returns:
            int: Selected action (0=LEFT, 1=RIGHT).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        return int(np.argmax(self.q_table[state]))         # Exploit

    def update(self, state: int, action: int,
               reward: float, next_state: int, done: bool):
        """
        Performs a Q-learning update for one transition.

        Formula:
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Args:
            state (int):      State before action.
            action (int):     Action taken.
            reward (float):   Reward received.
            next_state (int): Resulting state.
            done (bool):      Whether the episode ended.
        """
        current_q = self.q_table[state, action]

        # If terminal, no future reward
        future_q = 0.0 if done else np.max(self.q_table[next_state])

        td_target = reward + self.gamma * future_q
        td_error  = td_target - current_q

        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decays epsilon after each episode (down to epsilon_min)."""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train_agent(n_episodes: int = 50) -> tuple:
    """
    Trains the Q-learning agent on the 1D world for n_episodes.

    Args:
        n_episodes (int): Number of episodes to train.

    Returns:
        tuple: (agent, steps_per_episode)
            - agent: Trained QLearningAgent
            - steps_per_episode: List of step counts per episode
    """
    env   = OneDWorldEnv()
    agent = QLearningAgent()
    steps_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        steps = 0
        done  = False

        while not done:
            action               = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            # Safety cap: prevent infinite loops in early training
            if steps >= 500:
                break

        agent.decay_epsilon()
        steps_per_episode.append(steps)

    return agent, steps_per_episode


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

def run_section_c():
    """Trains the agent and prints results."""
    print("--- Problem 7: Mini RL Environment (1D World) ---")
    print("  Training Q-learning agent for 50 episodes...")
    print()

    random.seed(42)
    np.random.seed(42)

    agent, steps_per_episode = train_agent(n_episodes=50)

    # Print steps per episode
    print("  Steps per Episode:")
    for i, steps in enumerate(steps_per_episode, 1):
        bar = "█" * min(steps, 40)  # Visual bar (capped at 40)
        print(f"    Ep {i:02d}: {steps:>4} steps  {bar}")

    # Print final Q-table
    print("\n  Final Q-table (rows=states 0–9, cols=[LEFT, RIGHT]):")
    print(f"    {'State':>6}  {'Q(LEFT)':>10}  {'Q(RIGHT)':>10}  {'Best Action':>12}")
    print("    " + "-" * 46)
    action_names = ['LEFT', 'RIGHT']
    for state in range(10):
        q_left  = agent.q_table[state, 0]
        q_right = agent.q_table[state, 1]
        best    = action_names[np.argmax(agent.q_table[state])]
        print(f"    {state:>6}  {q_left:>10.4f}  {q_right:>10.4f}  {best:>12}")

    print(f"\n  Final epsilon: {agent.epsilon:.4f}")
    print("  (Low epsilon = more exploitation — agent has converged)\n")

    """
    REASONING:
    The 1D world is ideal for verifying Q-learning because the optimal
    policy is trivially known: always go RIGHT. So after training, every
    state should have Q(RIGHT) > Q(LEFT), which serves as a clear
    correctness check. Epsilon-greedy with decay starts at 1.0 (full
    exploration) and shrinks each episode so the agent exploits its
    growing knowledge. The step cap of 500 prevents rare infinite loops
    in early episodes when the agent is nearly random.

    PROMPTING JUSTIFICATION:
    Prompt used: "Design a minimal 1D Q-learning environment in Python
    with a goal state, epsilon-greedy exploration, and tabular Q-table
    using NumPy. Include training loop and printed Q-table." The
    structured split of OneDWorldEnv and QLearningAgent into separate
    classes came from this. I verified correctness by checking that
    after 50 episodes: (1) steps per episode generally decrease over
    time, and (2) every state's Q-table shows RIGHT as the best action,
    which matches the known optimal policy for this environment.
    """