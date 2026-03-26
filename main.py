"""
=============================================================
  Intelligent Systems Final Exam — Main Entry Point
  Author: Parejas, Arron Kian M.
  Description: Runs all sections of the project in sequence.
=============================================================
"""

from section_a import run_section_a
from section_b import run_section_b
from section_c import run_section_c


def main():
    """Main runner that executes all three sections."""
    print("=" * 60)
    print("   INTELLIGENT SYSTEMS FINAL EXAM PROJECT")
    print("=" * 60)

    print("\n>>> SECTION A: Robotics & Intelligent Agents\n")
    run_section_a()

    print("\n>>> SECTION B: Reinforcement Learning Fundamentals\n")
    run_section_b()

    print("\n>>> SECTION C: Capstone RL Project\n")
    run_section_c()

    print("\n" + "=" * 60)
    print("   ALL SECTIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()