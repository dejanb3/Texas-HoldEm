
"""
main.py — Entry Point
=====================
Launches the Texas Hold'em Q-Learning Lab GUI.

Usage::

    python main.py          # start the GUI
"""

from __future__ import annotations

from ui import PokerGUI


def main() -> None:
    """Create and run the application."""
    app = PokerGUI()
    app.run()


if __name__ == "__main__":
    main()
