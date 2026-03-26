"""
=============================================================
  SECTION A — Robotics & Intelligent Agents
  Problems: 1 (Grid Navigation), 2 (Sensor Fusion),
            3 (State Machine)
=============================================================
"""


# ─────────────────────────────────────────────
# PROBLEM 1: Robot Navigation Grid Simulator
# ─────────────────────────────────────────────

def navigate_robot(actions: list) -> tuple:
    """
    Simulates a robot moving on a 10x10 grid.

    The robot starts at position (0, 0) — the top-left corner.
    It accepts a list of direction strings and moves accordingly,
    clamping position within the valid 0–9 range on both axes.

    Args:
        actions (list): A list of strings — 'UP', 'DOWN', 'LEFT', 'RIGHT'

    Returns:
        tuple: Final (row, col) position of the robot.

    Assumptions:
        - Grid is 0-indexed, size 10x10 (rows 0–9, cols 0–9)
        - Invalid actions are silently ignored
        - 'UP' decreases row index (moves toward row 0)
    """
    row, col = 0, 0  # Start at top-left corner
    grid_size = 10

    # Direction map: each action changes (row, col)
    direction_map = {
        'UP':    (-1, 0),
        'DOWN':  (1, 0),
        'LEFT':  (0, -1),
        'RIGHT': (0, 1),
    }

    for action in actions:
        if action not in direction_map:
            print(f"  [WARN] Unknown action '{action}', skipping.")
            continue

        dr, dc = direction_map[action]
        new_row = row + dr
        new_col = col + dc

        # Clamp within grid boundaries (prevent out-of-bounds)
        row = max(0, min(grid_size - 1, new_row))
        col = max(0, min(grid_size - 1, new_col))

    return (row, col)


def run_problem_1():
    """Demonstrates the robot navigation simulator."""
    print("--- Problem 1: Robot Navigation ---")
    actions = ['RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'UP', 'LEFT', 'RIGHT', 'DOWN']
    final_pos = navigate_robot(actions)
    print(f"  Actions: {actions}")
    print(f"  Final Position: {final_pos}")

    # Edge case: robot tries to go out of bounds
    oob_actions = ['LEFT', 'LEFT', 'UP', 'UP', 'UP']
    oob_pos = navigate_robot(oob_actions)
    print(f"\n  Out-of-bounds test actions: {oob_actions}")
    print(f"  Final Position (clamped): {oob_pos}")

    """
    REASONING:
    A dictionary-based direction map is used instead of if/elif chains
    for cleanliness and extensibility. The clamping approach using
    max/min ensures the robot never leaves the grid without needing
    complex conditional logic. Starting at (0,0) is the standard
    convention for grid-based simulations.

    PROMPTING JUSTIFICATION:
    I used prompting to help clarify the best approach for grid clamping.
    The prompt was: "What is the cleanest way to prevent a robot from
    going out of bounds on a 2D grid in Python without using nested
    if statements?" The output suggested using min/max clamping which
    I verified manually by tracing through a few action sequences.
    This confirmed the function correctly handles edge and corner cases.
    """


# ─────────────────────────────────────────────
# PROBLEM 2: Sensor Fusion
# ─────────────────────────────────────────────

def sensor_fusion(left: float, center: float, right: float) -> str:
    """
    Determines robot steering direction based on three sensor readings.

    Each sensor returns a value between 0.0 (nothing detected) and
    1.0 (strong signal). A weighted sum is computed to determine
    the net steering direction.

    Weights:
        - LEFT sensor:   -1.0  (strong left signal → steer LEFT)
        - CENTER sensor:  0.0  (balanced → no lateral force)
        - RIGHT sensor:  +1.0  (strong right signal → steer RIGHT)

    Args:
        left (float): Left sensor reading (0.0 to 1.0)
        center (float): Center sensor reading (0.0 to 1.0)
        right (float): Right sensor reading (0.0 to 1.0)

    Returns:
        str: 'LEFT', 'RIGHT', or 'STRAIGHT'

    Assumptions:
        - A threshold of ±0.1 is used to avoid noise-triggered turns
        - Center sensor weight is 0 because it represents forward balance
    """
    # Weighted sum: right pulls positive, left pulls negative
    WEIGHT_LEFT   = -1.0
    WEIGHT_CENTER =  0.0
    WEIGHT_RIGHT  =  1.0
    THRESHOLD     =  0.1  # Dead zone to ignore minor sensor noise

    weighted_sum = (left * WEIGHT_LEFT) + \
                   (center * WEIGHT_CENTER) + \
                   (right * WEIGHT_RIGHT)

    if weighted_sum > THRESHOLD:
        return 'RIGHT'
    elif weighted_sum < -THRESHOLD:
        return 'LEFT'
    else:
        return 'STRAIGHT'


def run_problem_2():
    """Demonstrates sensor fusion with multiple test cases."""
    print("\n--- Problem 2: Sensor Fusion ---")

    test_cases = [
        (0.2, 0.5, 0.9, "Strong right signal"),
        (0.9, 0.5, 0.2, "Strong left signal"),
        (0.5, 0.8, 0.5, "Balanced sensors"),
        (0.0, 0.0, 0.0, "No signal at all"),
    ]

    for left, center, right, desc in test_cases:
        direction = sensor_fusion(left, center, right)
        print(f"  [{desc}] L={left}, C={center}, R={right} → {direction}")

    """
    REASONING:
    The weighted sum approach is chosen because it is a lightweight
    analog of how real sensor fusion systems work — combining multiple
    noisy signals into a single decision. The threshold of 0.1 acts as
    a dead zone to prevent the robot from jittering when sensors are
    nearly balanced. CENTER has weight 0 because it only indicates
    something is ahead, not a directional preference.

    PROMPTING JUSTIFICATION:
    I prompted: "How should I implement weighted sensor fusion for a
    simple robot with 3 directional sensors in Python, keeping it
    readable?" The suggestion to use named constants for weights and
    a threshold dead zone came from this process. I verified by testing
    symmetric inputs (equal left and right) which consistently returned
    STRAIGHT, confirming correct logic.
    """


# ─────────────────────────────────────────────
# PROBLEM 3: State Machine
# ─────────────────────────────────────────────

def get_next_state(current_state: str,
                   object_detected: bool,
                   distance: float) -> str:
    """
    Determines the next robot state based on current state and inputs.

    States:
        SEARCH   → Robot is scanning for an object
        APPROACH → Object found, robot is moving toward it
        GRASP    → Robot is close enough to grab the object

    Transitions:
        SEARCH   → APPROACH  : if object_detected is True
        APPROACH → GRASP     : if distance <= 0.5 (close enough)
        APPROACH → SEARCH    : if object_detected becomes False
        GRASP    → SEARCH    : always resets after grasping

    Args:
        current_state (str): One of 'SEARCH', 'APPROACH', 'GRASP'
        object_detected (bool): Whether an object is currently visible
        distance (float): Estimated distance to the object (0.0 = contact)

    Returns:
        str: The next state.
    """
    GRASP_DISTANCE = 0.5  # Units: meters (assumed)

    if current_state == 'SEARCH':
        if object_detected:
            return 'APPROACH'
        return 'SEARCH'

    elif current_state == 'APPROACH':
        if not object_detected:
            return 'SEARCH'  # Lost the object
        if distance <= GRASP_DISTANCE:
            return 'GRASP'
        return 'APPROACH'

    elif current_state == 'GRASP':
        # After grasping, return to search for next object
        return 'SEARCH'

    else:
        raise ValueError(f"Unknown state: '{current_state}'")


def run_problem_3():
    """Simulates a sequence of state machine transitions."""
    print("\n--- Problem 3: State Machine ---")

    # Simulate a scenario: robot searches, finds object, approaches, grasps
    scenario = [
        # (object_detected, distance, description)
        (False, 5.0, "Scanning — no object"),
        (True,  3.0, "Object spotted — far away"),
        (True,  1.5, "Approaching — still far"),
        (True,  0.4, "Close enough to grasp"),
        (False, 0.0, "Post-grasp reset"),
    ]

    state = 'SEARCH'
    print(f"  Initial State: {state}")

    for obj_det, dist, desc in scenario:
        next_state = get_next_state(state, obj_det, dist)
        print(f"  [{desc}] detected={obj_det}, dist={dist} "
              f"→ {state} → {next_state}")
        state = next_state

    """
    REASONING:
    A function-based state machine is cleaner and more testable than
    a class-based one for this scale of problem. Each state is handled
    by its own conditional block, making transitions explicit and easy
    to read. The GRASP_DISTANCE constant is named clearly to convey
    intent. The GRASP state always resets to SEARCH, modeling a real
    pick-and-place loop.

    PROMPTING JUSTIFICATION:
    Prompt used: "How do I implement a clean finite state machine in
    Python using only functions, for a robot with SEARCH, APPROACH,
    and GRASP states?" The key insight I got was to handle each state
    in its own block and raise ValueError for unknown states. I verified
    by running the scenario and checking that each transition matched
    the expected behavior described in the problem.
    """


def run_section_a():
    """Runs all Section A problems."""
    run_problem_1()
    run_problem_2()
    run_problem_3()