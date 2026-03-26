"""
=============================================================
  utils.py — Shared Utility Functions
=============================================================
"""


def print_section_header(title: str):
    """Prints a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamps a value between min_val and max_val.

    Args:
        value: The input value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        float: Clamped value.
    """
    return max(min_val, min(max_val, value))
