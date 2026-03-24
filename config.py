"""
Configuration & Constants
=========================
Central configuration module for the Texas Hold'em Q-Learning Lab.

Houses **every** tuneable constant and colour token so that the rest of
the codebase is free of magic values.

Sections
--------
1. Poker rules (ranks, suits, fixed scenario)
2. MDP defaults (stacks, pot, opponent aggression)
3. Q-Learning hyperparameter defaults
4. GUI / theme palette
"""

from __future__ import annotations

from typing import Dict, Final, List


# ============================================================================
# 1. Poker Rules
# ============================================================================

RANKS: Final[List[str]] = [
    "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"
]
SUITS: Final[List[str]] = ["h", "d", "c", "s"]

SUIT_SYMBOLS: Final[Dict[str, str]] = {
    "h": "♥", "d": "♦", "c": "♣", "s": "♠",
}

HAND_NAMES: Final[Dict[int, str]] = {
    10: "Royal Flush",
    9:  "Straight Flush",
    8:  "Four of a Kind",
    7:  "Full House",
    6:  "Flush",
    5:  "Straight",
    4:  "Three of a Kind",
    3:  "Two Pair",
    2:  "One Pair",
    1:  "High Card",
}


# ============================================================================
# 2. MDP / Environment Defaults
# ============================================================================

# Default scenario (post-flop heads-up)
HERO_CARDS: Final[List[str]] = ["8h", "9h"]       # 8♥ 9♥
FLOP_CARDS: Final[List[str]] = ["Jh", "Qh", "2c"]  # J♥ Q♥ 2♣

# When True, hero cards and flop are randomised every hand during training
STOCHASTIC: Final[bool] = True

INITIAL_STACK: Final[int] = 150
INITIAL_POT: Final[int] = 100

OPPONENT_AGGRESSION: Final[float] = 0.3
OPPONENT_FOLD_PROB: Final[float] = 0.10

ACTIONS: Final[List[str]] = ["fold", "call", "raise_50", "raise_100", "all_in"]


# ============================================================================
# 3. Q-Learning Hyperparameter Defaults
# ============================================================================

DEFAULT_LEARNING_RATE: Final[float] = 0.1     # α
DEFAULT_DISCOUNT_FACTOR: Final[float] = 0.95  # γ
DEFAULT_EPSILON: Final[float] = 0.2           # ε
DEFAULT_EPISODES: Final[int] = 5000
DEFAULT_VERBOSE_EVERY: Final[int] = 100


# ============================================================================
# 4. GUI / Theme Palette
# ============================================================================

# -- CustomTkinter appearance --
CTK_APPEARANCE_MODE: Final[str] = "dark"
CTK_COLOUR_THEME: Final[str] = "blue"

# -- Poker table --
TABLE_GREEN: Final[str] = "#1a472a"
FELT_GREEN: Final[str] = "#2d5016"

# -- Cards --
CARD_BG: Final[str] = "#ffffff"
CARD_RED: Final[str] = "#d32f2f"
CARD_BLACK: Final[str] = "#212121"
CARD_BACK_COLOUR: Final[str] = "#1565c0"

SUIT_COLOURS: Final[Dict[str, str]] = {
    "h": CARD_RED, "d": CARD_RED,
    "c": CARD_BLACK, "s": CARD_BLACK,
}

# -- Accent colours --
GOLD: Final[str] = "#ffd700"
ACCENT_BLUE: Final[str] = "#1e88e5"
ACCENT_RED: Final[str] = "#e53935"
ACCENT_GREEN: Final[str] = "#43a047"

# -- Matplotlib figure palette --
FIG_FACECOLOR: Final[str] = "#1e1e2e"
AX_FACECOLOR: Final[str] = "#1e1e2e"
GRID_COLOUR: Final[str] = "#37474f"
TITLE_COLOUR: Final[str] = "#e0e0e0"
LABEL_COLOUR: Final[str] = "#90a4ae"
TICK_COLOUR: Final[str] = "#78909c"
LEGEND_BG: Final[str] = "#263238"
LEGEND_EDGE: Final[str] = "#37474f"
LEGEND_TEXT: Final[str] = "#b0bec5"

# Win-Rate plot
WR_LINE_COLOUR: Final[str] = "#69f0ae"
WR_FILL_ALPHA: Final[float] = 0.15
BASELINE_COLOUR: Final[str] = "#546e7a"

# Reward plot
RW_RAW_COLOUR: Final[str] = "#42a5f5"
RW_MA_COLOUR: Final[str] = "#ef5350"
RW_FILL_ALPHA: Final[float] = 0.12

# Heatmap / Q-Table shared colourmap stops
CMAP_STOPS: Final[List[str]] = [
    "#ef5350", "#37474f", "#1e1e2e", "#37474f", "#69f0ae",
]

# -- Action bar display (thought process panel) --
ACTION_DISPLAY: Final[Dict[str, tuple]] = {
    "fold":      ("Fold",       "#ef5350"),
    "call":      ("Call",       "#66bb6a"),
    "raise_50":  ("Raise $50",  "#29b6f6"),
    "raise_100": ("Raise $100", "#ffa726"),
    "all_in":    ("All-In",     "#ab47bc"),
}

# -- Window defaults --
WINDOW_TITLE: Final[str] = "Texas Hold'em · Q-Learning Lab"
WINDOW_GEOMETRY: Final[str] = "1400x900"
WINDOW_MIN_SIZE: Final[tuple] = (1200, 800)

# -- Multi-window layout --
TABLE_WINDOW_TITLE: Final[str] = "Poker Table"
TABLE_WINDOW_GEOMETRY: Final[str] = "1060x920"
TABLE_WINDOW_MIN_SIZE: Final[tuple] = (900, 800)

ANALYSIS_WINDOW_TITLE: Final[str] = "Train and Analyse"
ANALYSIS_WINDOW_GEOMETRY: Final[str] = "680x920"
ANALYSIS_WINDOW_MIN_SIZE: Final[tuple] = (550, 700)
