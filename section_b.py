"""
=============================================================
  SECTION B — Reinforcement Learning Fundamentals
  Problems: 4 (Q-Learning Update), 5 (Softmax Selection),
            6 (Policy Network)
=============================================================
"""

import math
import random
import numpy as np


# ─────────────────────────────────────────────
# PROBLEM 4: Q-Learning Update
# ─────────────────────────────────────────────

def q_learning_update(q_table: dict,
                      state: int,
                      action: int,
                      reward: float,
                      next_state: int,
                      alpha: float = 0.1,
                      gamma: float = 0.9) -> dict:
    """
    Applies one step of the Q-learning update rule to the Q-table.

    Q-learning formula:
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]

    Args:
        q_table (dict): Dictionary keyed by (state, action) pairs.
        state (int): Current state index.
        action (int): Action taken.
        reward (float): Reward received after taking the action.
        next_state (int): The resulting next state.
        alpha (float): Learning rate (0 < α ≤ 1). Default 0.1.
        gamma (float): Discount factor (0 ≤ γ ≤ 1). Default 0.9.

    Returns:
        dict: Updated Q-table.

    Assumptions:
        - Q-table uses (state, action) tuples as keys
        - Number of actions is inferred from existing keys
        - Missing Q-values default to 0.0
    """
    # Get all known actions for the next state
    known_actions = [k[1] for k in q_table if k[0] == next_state]

    if known_actions:
        # Best future Q-value: max over all actions from next_state
        max_future_q = max(q_table.get((next_state, a), 0.0)
                           for a in known_actions)
    else:
        max_future_q = 0.0  # Terminal or unknown state

    current_q = q_table.get((state, action), 0.0)

    # Core Q-learning update formula
    td_error = reward + gamma * max_future_q - current_q
    q_table[(state, action)] = current_q + alpha * td_error

    return q_table


def run_problem_4():
    """Demonstrates the Q-learning update on a small Q-table."""
    print("--- Problem 4: Q-Learning Update ---")

    # Initialize a small Q-table for states 0–2, actions 0–1
    q_table = {}
    for s in range(3):
        for a in range(2):
            q_table[(s, a)] = 0.0

    print("  Initial Q-table:")
    for key, val in q_table.items():
        print(f"    Q{key} = {val:.4f}")

    # Simulate a single update: agent in state 0 took action 1,
    # got reward +10, landed in state 2
    q_table = q_learning_update(q_table, state=0, action=1,
                                 reward=10.0, next_state=2)

    print("\n  After update (state=0, action=1, reward=10, next=2):")
    for key, val in q_table.items():
        print(f"    Q{key} = {val:.4f}")

    """
    REASONING:
    The dictionary Q-table is chosen for clarity and flexibility over
    a NumPy array, since states/actions can be arbitrary. The TD error
    is computed explicitly to make the formula visible and verifiable.
    defaulting missing Q-values to 0.0 is standard practice for
    initializing optimistic or neutral Q-tables.

    PROMPTING JUSTIFICATION:
    Prompt used: "Implement the Q-learning update formula in Python
    using a dictionary as the Q-table, with clear variable names that
    match the mathematical formula." This helped me structure the
    variables (td_error, max_future_q) to directly mirror the Bellman
    equation. I verified correctness by manually computing the expected
    Q(0,1) = 0 + 0.1 * [10 + 0.9 * 0 - 0] = 1.0 and checking the output.
    """


# ─────────────────────────────────────────────
# PROBLEM 5: Softmax Action Selection
# ─────────────────────────────────────────────

def softmax(q_values: list, temperature: float = 1.0) -> list:
    """
    Computes softmax probabilities over a list of Q-values.

    The temperature parameter τ controls exploration:
        - High τ → uniform probabilities (more exploration)
        - Low τ  → greedy-like (more exploitation)

    Formula:
        P(a) = exp(Q(a) / τ) / Σ exp(Q(i) / τ)

    Args:
        q_values (list): List of Q-values for each action.
        temperature (float): Temperature τ > 0. Default 1.0.

    Returns:
        list: Probability distribution over actions (sums to 1.0).
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Scale Q-values by temperature
    scaled = [q / temperature for q in q_values]

    # Numerical stability: subtract max before exponentiation
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    total = sum(exp_vals)

    return [e / total for e in exp_vals]


def select_action_softmax(q_values: list, temperature: float = 1.0) -> int:
    """
    Selects an action by sampling from the softmax distribution.

    Args:
        q_values (list): Q-values for each available action.
        temperature (float): Exploration temperature.

    Returns:
        int: Index of the selected action.
    """
    probs = softmax(q_values, temperature)

    # Weighted random sampling
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1  # Fallback for floating point edge case


def run_problem_5():
    """Demonstrates softmax selection with different temperatures."""
    print("\n--- Problem 5: Softmax Action Selection ---")

    q_values = [1.0, 2.5, 0.3]
    print(f"  Q-values: {q_values}")

    for tau in [0.1, 1.0, 5.0]:
        probs = softmax(q_values, temperature=tau)
        chosen = select_action_softmax(q_values, temperature=tau)
        print(f"  τ={tau:4.1f} → Probs: {[f'{p:.3f}' for p in probs]}"
              f"  | Sampled action: {chosen}")

    """
    REASONING:
    Subtracting the max value before exponentiation is a standard
    numerical stability trick — without it, large Q-values cause
    overflow in math.exp(). The temperature parameter makes this
    implementation suitable for both exploration (high τ) and
    exploitation (low τ). Weighted random sampling is preferred over
    numpy.random.choice for pure Python compliance.

    PROMPTING JUSTIFICATION:
    Prompt used: "How do I implement softmax with a temperature
    parameter in pure Python, including numerical stability?" The
    max-subtraction trick was the key insight from this prompt.
    I verified by checking that probabilities sum to 1.0 across all
    temperature values and that low temperature consistently selected
    the highest Q-value action.
    """


# ─────────────────────────────────────────────
# PROBLEM 6: Policy Network (NumPy only)
# ─────────────────────────────────────────────

class PolicyNetwork:
    """
    A simple 2-layer policy network implemented with NumPy only.

    Architecture:
        Input  → Layer 1 (4 → 16, ReLU) → Layer 2 (16 → 3, Linear)

    The network outputs 3 raw logits which can be passed to softmax
    for action selection in a policy gradient algorithm.
    """

    def __init__(self, input_size: int = 4,
                 hidden_size: int = 16,
                 output_size: int = 3):
        """
        Initializes weights using He initialization (good for ReLU).

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Number of output logits (actions).
        """
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # He initialization: std = sqrt(2 / fan_in)
        self.W1 = np.random.randn(input_size, hidden_size) * \
                  np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, output_size) * \
                  np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)."""
        return np.maximum(0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x (np.ndarray): Input vector of shape (input_size,).

        Returns:
            np.ndarray: Output logits of shape (output_size,).
        """
        # Layer 1: linear + ReLU
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)

        # Layer 2: linear (no activation — raw logits)
        z2 = np.dot(a1, self.W2) + self.b2

        return z2

    def get_action_probs(self, x: np.ndarray,
                         temperature: float = 1.0) -> np.ndarray:
        """
        Returns softmax probabilities over actions given an input state.

        Args:
            x (np.ndarray): Input state vector.
            temperature (float): Softmax temperature for exploration.

        Returns:
            np.ndarray: Probability distribution over actions.
        """
        logits = self.forward(x)
        scaled = logits / temperature
        # Numerically stable softmax
        exp_vals = np.exp(scaled - np.max(scaled))
        return exp_vals / exp_vals.sum()


def run_problem_6():
    """Demonstrates the policy network with a sample input."""
    print("\n--- Problem 6: Policy Network ---")

    np.random.seed(42)  # For reproducibility
    net = PolicyNetwork(input_size=4, hidden_size=16, output_size=3)

    # Sample input: simulated state vector (e.g., CartPole-like)
    sample_state = np.array([0.1, -0.3, 0.05, 0.2])

    logits = net.forward(sample_state)
    probs  = net.get_action_probs(sample_state, temperature=1.0)

    print(f"  Input state:   {sample_state}")
    print(f"  Output logits: {np.round(logits, 4)}")
    print(f"  Action probs:  {np.round(probs, 4)}")
    print(f"  Selected action: {np.argmax(probs)} (greedy)")

    """
    REASONING:
    He initialization is used instead of random normal or Xavier
    because ReLU activations work best with He scaling — it prevents
    vanishing/exploding gradients in deeper networks. The output layer
    returns raw logits (no softmax) to allow flexibility: logits can
    be used directly in cross-entropy loss or passed to a separate
    softmax for inference. NumPy is sufficient for a forward-pass-only
    policy network at this scale.

    PROMPTING JUSTIFICATION:
    Prompt used: "Build a 2-layer neural network policy in NumPy with
    ReLU hidden layer, returning logits. Use He initialization and
    include a softmax method." The structured layer naming (W1, b1, W2,
    b2) and the separation of forward() from get_action_probs() came
    from this prompting session. I verified by checking output shape
    (3 logits), confirming probs sum to 1.0, and checking that ReLU
    zeroed negative activations correctly on a test input.
    """


def run_section_b():
    """Runs all Section B problems."""
    run_problem_4()
    run_problem_5()
    run_problem_6()