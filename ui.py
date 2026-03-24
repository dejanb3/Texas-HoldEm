"""
Texas Hold'em Q-Learning -- CustomTkinter GUI  (View)
=====================================================
Multi-window architecture:

* **Poker Table** (primary) -- card display, action buttons, game controls.
* **Train and Analyse** (secondary) -- training hyperparameters, progress,
  and four analytics tabs (Win Rate, Reward, Q-Table Heatmap, Q-Table grid).

Atomic-Action Protocol
----------------------
The UI never lets the environment execute more than ONE action per call.

* ``env.step_opponent()`` -- opponent acts once, returns.
* ``env.step(action)``    -- hero acts once, returns.

When both players have acted and bets match on a street
(``info["street_settled"]``), the game STOPS and waits for the
**"Deal Next Card"** button (flop/turn) or proceeds directly to
showdown (river):

* In **Manual** mode the player clicks it.
* In **Watch AI** mode the UI auto-clicks it after a delay.

This eliminates all double-move and auto-transition bugs.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, Dict, List, Optional

import customtkinter as ctk
import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

from config import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACTION_DISPLAY,
    ANALYSIS_WINDOW_GEOMETRY,
    ANALYSIS_WINDOW_MIN_SIZE,
    ANALYSIS_WINDOW_TITLE,
    AX_FACECOLOR,
    BASELINE_COLOUR,
    CARD_BG,
    CARD_BACK_COLOUR,
    CARD_BLACK,
    CMAP_STOPS,
    CTK_APPEARANCE_MODE,
    CTK_COLOUR_THEME,
    FELT_GREEN,
    FIG_FACECOLOR,
    GOLD,
    GRID_COLOUR,
    LABEL_COLOUR,
    LEGEND_BG,
    LEGEND_EDGE,
    LEGEND_TEXT,
    RW_FILL_ALPHA,
    RW_MA_COLOUR,
    RW_RAW_COLOUR,
    SUIT_COLOURS,
    TABLE_GREEN,
    TABLE_WINDOW_GEOMETRY,
    TABLE_WINDOW_MIN_SIZE,
    TABLE_WINDOW_TITLE,
    TICK_COLOUR,
    TITLE_COLOUR,
    WR_FILL_ALPHA,
    WR_LINE_COLOUR,
)
from environment import Card, GameState, HandEvaluator, PokerEnv, StepResult
from agent import QLearningAgent

# ============================================================================
# Theme
# ============================================================================

ctk.set_appearance_mode(CTK_APPEARANCE_MODE)
ctk.set_default_color_theme(CTK_COLOUR_THEME)


# ============================================================================
# Card Widget
# ============================================================================

class CardWidget(ctk.CTkFrame):
    """Visual playing-card widget with optional face-down state."""

    WIDTH = 70
    HEIGHT = 100

    def __init__(
        self,
        master: Any,
        card: Optional[Card] = None,
        face_up: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            master,
            width=self.WIDTH,
            height=self.HEIGHT,
            corner_radius=8,
            fg_color=CARD_BG if face_up else CARD_BACK_COLOUR,
            border_width=2,
            border_color="#bdbdbd",
            **kwargs,
        )
        self.pack_propagate(False)
        self.grid_propagate(False)

        if face_up and card is not None:
            colour = SUIT_COLOURS.get(card.suit, CARD_BLACK)
            lbl = ctk.CTkLabel(
                self,
                text=card.symbol,
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color=colour,
                fg_color="transparent",
            )
            lbl.place(relx=0.5, rely=0.5, anchor="center")
        elif not face_up:
            lbl = ctk.CTkLabel(
                self,
                text="?",
                font=ctk.CTkFont(size=28, weight="bold"),
                text_color="#ffffff",
                fg_color="transparent",
            )
            lbl.place(relx=0.5, rely=0.5, anchor="center")


# ============================================================================
# Poker GUI  --  Multi-window controller
# ============================================================================

class PokerGUI:
    """Two-window application: Poker Table + Analysis & Training."""

    def __init__(self) -> None:
        self.root = ctk.CTk()
        self.root.title(TABLE_WINDOW_TITLE)
        self.root.geometry(TABLE_WINDOW_GEOMETRY)
        self.root.minsize(*TABLE_WINDOW_MIN_SIZE)

        self.env = PokerEnv()
        self.agent = QLearningAgent(actions=self.env.actions)

        self.game_active: bool = False
        self.training_active: bool = False
        self.current_state: Optional[GameState] = None
        self._card_widgets: List[CardWidget] = []
        self._action_log: List[str] = []
        self._next_hand_mode: Optional[str] = None  # "manual" or "ai"
        self._pending_timers: List[str] = []  # tracked after() IDs
        self._processing: bool = False  # mutual-exclusion guard

        self._build_table_window()

        self.analysis_win = ctk.CTkToplevel(self.root)
        self.analysis_win.title(ANALYSIS_WINDOW_TITLE)
        self.analysis_win.geometry(ANALYSIS_WINDOW_GEOMETRY)
        self.analysis_win.minsize(*ANALYSIS_WINDOW_MIN_SIZE)
        self.analysis_win.after(50, self._position_analysis_window)
        self.analysis_win.protocol("WM_DELETE_WINDOW", self._toggle_analysis)

        self._build_analysis_window()

    # ================================================================
    # Window helpers
    # ================================================================

    def _position_analysis_window(self) -> None:
        try:
            x = self.root.winfo_x() + self.root.winfo_width() + 8
            y = self.root.winfo_y()
            self.analysis_win.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _switch_to_analysis(self) -> None:
        self.analysis_win.deiconify()
        self.analysis_win.lift()
        self.analysis_win.focus_force()

    def _switch_to_table(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _toggle_analysis(self) -> None:
        if self.analysis_win.winfo_viewable():
            self.analysis_win.withdraw()
        else:
            self._switch_to_analysis()

    # ================================================================
    # Timer management -- prevents ghost callbacks
    # ================================================================

    def _schedule(self, ms: int, func) -> None:
        """Schedule *func* after *ms* milliseconds, tracking the ID."""
        tid = self.root.after(ms, func)
        self._pending_timers.append(tid)

    def _cancel_pending(self) -> None:
        """Cancel every pending scheduled callback."""
        for tid in self._pending_timers:
            try:
                self.root.after_cancel(tid)
            except Exception:
                pass
        self._pending_timers.clear()

    # ================================================================
    # Primary window -- Poker Table
    # ================================================================

    def _build_table_window(self) -> None:
        root = self.root
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        table = ctk.CTkFrame(root, fg_color=TABLE_GREEN, corner_radius=12)
        table.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        table.grid_rowconfigure(2, weight=1)
        table.grid_columnconfigure(0, weight=3)
        table.grid_columnconfigure(1, weight=2)

        # ---- Right-side info panel ----
        right_panel = ctk.CTkFrame(table, fg_color="transparent")
        right_panel.grid(row=0, column=1, rowspan=8, sticky="nsew",
                         padx=(4, 12), pady=12)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        # AI Thought Process
        self.thought_frame = ctk.CTkFrame(
            right_panel, fg_color="#1a2332", corner_radius=10,
            border_width=1, border_color="#2d4a5e",
        )
        self.thought_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))

        thought_header = ctk.CTkFrame(self.thought_frame, fg_color="transparent")
        thought_header.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            thought_header, text="AI Thought Process",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#80cbc4",
        ).pack(side="left")
        self.thought_state_label = ctk.CTkLabel(
            thought_header, text="",
            font=ctk.CTkFont(size=11), text_color="#546e7a",
        )
        self.thought_state_label.pack(side="right")

        self.thought_bars_frame = ctk.CTkFrame(
            self.thought_frame, fg_color="transparent",
        )
        self.thought_bars_frame.pack(fill="x", padx=12, pady=(4, 12))

        self._thought_bar_widgets: Dict[str, Dict[str, Any]] = {}
        for action in self.env.actions:
            display_name, colour = ACTION_DISPLAY.get(action, (action, "#78909c"))
            bar_row = ctk.CTkFrame(self.thought_bars_frame, fg_color="transparent")
            bar_row.pack(fill="x", pady=1)
            bar_row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(
                bar_row, text=display_name, width=80,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#cfd8dc", anchor="w",
            ).grid(row=0, column=0, padx=(0, 6), sticky="w")

            bar = ctk.CTkProgressBar(
                bar_row, height=18, corner_radius=4,
                progress_color=colour, fg_color="#263238",
                border_color="#37474f", border_width=1,
            )
            bar.grid(row=0, column=1, sticky="ew", padx=(0, 6))
            bar.set(0)

            val_lbl = ctk.CTkLabel(
                bar_row, text="--", width=70,
                font=ctk.CTkFont(family="Courier", size=11, weight="bold"),
                text_color="#90a4ae", anchor="e",
            )
            val_lbl.grid(row=0, column=2, sticky="e")

            badge_lbl = ctk.CTkLabel(
                bar_row, text="", width=24,
                font=ctk.CTkFont(size=12), text_color=GOLD,
            )
            badge_lbl.grid(row=0, column=3, padx=(2, 0))

            self._thought_bar_widgets[action] = {
                "bar": bar, "val": val_lbl, "badge": badge_lbl, "colour": colour,
            }

        # Action Log
        self.log_frame = ctk.CTkFrame(
            right_panel, fg_color="#1a2332", corner_radius=10,
            border_width=1, border_color="#2d4a5e",
        )
        self.log_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        ctk.CTkLabel(
            self.log_frame, text="Action Log",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#80cbc4",
        ).pack(padx=10, pady=(10, 4), anchor="w")

        self.log_label = ctk.CTkLabel(
            self.log_frame, text="No actions yet.",
            font=ctk.CTkFont(family="Courier", size=12),
            text_color="#546e7a", justify="left", anchor="nw",
        )
        self.log_label.pack(fill="both", expand=True, padx=12, pady=(2, 10))

        # Title
        ctk.CTkLabel(
            table, text="Texas Hold'em",
            font=ctk.CTkFont(size=28, weight="bold"), text_color=GOLD,
        ).grid(row=0, column=0, pady=(18, 8))

        # Opponent area
        self.opp_frame = ctk.CTkFrame(table, fg_color="transparent")
        self.opp_frame.grid(row=1, column=0, pady=5)
        self.opp_label = ctk.CTkLabel(
            self.opp_frame, text="Opponent",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#e0e0e0",
        )
        self.opp_label.pack(pady=(0, 4))
        self.opp_cards_frame = ctk.CTkFrame(self.opp_frame, fg_color="transparent")
        self.opp_cards_frame.pack()
        self.opp_stack_label = ctk.CTkLabel(
            self.opp_frame, text="Stack: $150",
            font=ctk.CTkFont(size=14), text_color="#ef9a9a",
        )
        self.opp_stack_label.pack(pady=(4, 6))
        self._show_opp_cards(face_up=False)

        # Community cards
        self.community_frame = ctk.CTkFrame(
            table, fg_color="#1b5e20", corner_radius=10,
        )
        self.community_frame.grid(row=2, column=0, pady=12, padx=50, sticky="ew")
        ctk.CTkLabel(
            self.community_frame, text="Community Cards",
            font=ctk.CTkFont(size=14), text_color="#a5d6a7",
        ).pack(pady=(10, 4))
        self.board_frame = ctk.CTkFrame(
            self.community_frame, fg_color="transparent",
        )
        self.board_frame.pack(pady=(0, 12))

        # Total Pot (centered)
        self.total_pot_label = ctk.CTkLabel(
            table, text="Total Pot: $100",
            font=ctk.CTkFont(size=26, weight="bold"), text_color=GOLD,
        )
        self.total_pot_label.grid(row=3, column=0, pady=(10, 2))

        # Pot detail
        self.info_frame = ctk.CTkFrame(table, fg_color="transparent")
        self.info_frame.grid(row=4, column=0, pady=(0, 4))
        self.pot_label = ctk.CTkLabel(
            self.info_frame,
            text="Hero invested: $0  |  Opp invested: $0",
            font=ctk.CTkFont(size=14), text_color="#b0bec5",
        )
        self.pot_label.pack()
        self.street_label = ctk.CTkLabel(
            self.info_frame, text="",
            font=ctk.CTkFont(size=15), text_color="#b0bec5",
        )
        self.street_label.pack()

        # Result / message banner
        self.msg_frame = ctk.CTkFrame(
            table, fg_color="transparent", corner_radius=10,
        )
        self.msg_frame.grid(row=4, column=0, sticky="ew", padx=40, pady=(2, 2))
        self.msg_frame.grid_columnconfigure(0, weight=1)
        self.msg_label = ctk.CTkLabel(
            self.msg_frame, text="",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#ffcc80", wraplength=700,
        )
        self.msg_label.grid(row=0, column=0, pady=10, padx=16)

        # "Deal Next Card" button (hidden by default)
        self.deal_btn = ctk.CTkButton(
            self.msg_frame, text="Deal Next Card", width=180, height=38,
            fg_color="#00897b", hover_color="#00695c",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._deal_next_card,
        )

        # "Next Hand" button (hidden by default)
        self.next_hand_btn = ctk.CTkButton(
            self.msg_frame, text="Next Hand", width=140, height=36,
            fg_color=ACCENT_BLUE, hover_color="#1565c0",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._next_hand,
        )

        # Hero cards
        self.hero_frame = ctk.CTkFrame(
            table, fg_color=FELT_GREEN, corner_radius=10,
        )
        self.hero_frame.grid(row=5, column=0, pady=8, padx=50, sticky="ew")
        ctk.CTkLabel(
            self.hero_frame, text="Hero (You)",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff",
        ).pack(pady=(10, 4))
        self.hero_cards_frame = ctk.CTkFrame(
            self.hero_frame, fg_color="transparent",
        )
        self.hero_cards_frame.pack()
        self.hero_stack_label = ctk.CTkLabel(
            self.hero_frame, text="Stack: $150",
            font=ctk.CTkFont(size=14), text_color="#c8e6c9",
        )
        self.hero_stack_label.pack(pady=(4, 10))
        self._show_hero_cards()

        # Action buttons
        self.btn_frame = ctk.CTkFrame(table, fg_color="transparent")
        self.btn_frame.grid(row=6, column=0, pady=10)

        self.fold_btn = ctk.CTkButton(
            self.btn_frame, text="Fold", width=100, height=42,
            fg_color=ACCENT_RED, hover_color="#c62828",
            command=lambda: self._player_action("fold"),
        )
        self.fold_btn.grid(row=0, column=0, padx=4)

        self.call_btn = ctk.CTkButton(
            self.btn_frame, text="Call", width=100, height=42,
            fg_color=ACCENT_GREEN, hover_color="#2e7d32",
            command=lambda: self._player_action("call"),
        )
        self.call_btn.grid(row=0, column=1, padx=4)

        self.raise50_btn = ctk.CTkButton(
            self.btn_frame, text="Raise $50", width=110, height=42,
            fg_color="#29b6f6", hover_color="#0288d1",
            command=lambda: self._player_action("raise_50"),
        )
        self.raise50_btn.grid(row=0, column=2, padx=4)

        self.raise100_btn = ctk.CTkButton(
            self.btn_frame, text="Raise $100", width=120, height=42,
            fg_color="#ff8f00", hover_color="#e65100",
            command=lambda: self._player_action("raise_100"),
        )
        self.raise100_btn.grid(row=0, column=3, padx=4)

        self.raise150_btn = ctk.CTkButton(
            self.btn_frame, text="All-In $150", width=120, height=42,
            fg_color="#d50000", hover_color="#b71c1c",
            command=lambda: self._player_action("raise_150"),
        )
        self.raise150_btn.grid(row=0, column=4, padx=4)

        self._disable_actions()

        # Control bar
        ctrl = ctk.CTkFrame(table, fg_color="transparent")
        ctrl.grid(row=7, column=0, pady=(6, 14))

        self.new_game_btn = ctk.CTkButton(
            ctrl, text="New Game (Manual)", width=190, height=38,
            fg_color=ACCENT_BLUE, hover_color="#1565c0",
            command=self._start_manual,
        )
        self.new_game_btn.grid(row=0, column=0, padx=6)

        self.ai_play_btn = ctk.CTkButton(
            ctrl, text="Watch AI Play", width=190, height=38,
            fg_color="#6a1b9a", hover_color="#4a148c",
            command=self._watch_ai,
        )
        self.ai_play_btn.grid(row=0, column=1, padx=6)

        self.toggle_btn = ctk.CTkButton(
            ctrl, text="Train & Analyse", width=190, height=38,
            fg_color="#37474f", hover_color="#455a64",
            command=self._switch_to_analysis,
        )
        self.toggle_btn.grid(row=0, column=2, padx=6)

    # ================================================================
    # Secondary window -- Analysis & Training
    # ================================================================

    def _build_analysis_window(self) -> None:
        win = self.analysis_win
        win.grid_rowconfigure(2, weight=1)
        win.grid_columnconfigure(0, weight=1)

        nav_bar = ctk.CTkFrame(win, fg_color="transparent")
        nav_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 0))
        nav_bar.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            nav_bar, text="Back to Poker Table", height=34,
            fg_color=TABLE_GREEN, hover_color=FELT_GREEN,
            font=ctk.CTkFont(size=13, weight="bold"), text_color=GOLD,
            command=self._switch_to_table,
        ).grid(row=0, column=0, sticky="ew")

        train_outer = ctk.CTkFrame(win, fg_color="transparent")
        train_outer.grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 6))
        train_outer.grid_columnconfigure(0, weight=1)

        train_ctl = ctk.CTkFrame(train_outer, fg_color="transparent")
        train_ctl.grid(row=0, column=0)

        ctk.CTkLabel(
            train_ctl, text="Q-Learning Training",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, pady=(0, 10))

        ctk.CTkLabel(train_ctl, text="Episodes:").grid(row=1, column=0, padx=4)
        self.ep_var = ctk.StringVar(value="5000")
        ctk.CTkEntry(train_ctl, textvariable=self.ep_var, width=80).grid(
            row=1, column=1, padx=4, sticky="w",
        )

        ctk.CTkLabel(train_ctl, text="a:").grid(row=1, column=2, padx=(12, 2))
        self.lr_var = ctk.StringVar(value="0.10")
        ctk.CTkEntry(train_ctl, textvariable=self.lr_var, width=60).grid(
            row=1, column=3, padx=2,
        )

        ctk.CTkLabel(train_ctl, text="g:").grid(row=2, column=0, padx=4, pady=4)
        self.gamma_var = ctk.StringVar(value="0.95")
        ctk.CTkEntry(train_ctl, textvariable=self.gamma_var, width=60).grid(
            row=2, column=1, padx=4, sticky="w",
        )

        ctk.CTkLabel(train_ctl, text="e:").grid(row=2, column=2, padx=(12, 2))
        self.eps_var = ctk.StringVar(value="0.20")
        ctk.CTkEntry(train_ctl, textvariable=self.eps_var, width=60).grid(
            row=2, column=3, padx=2,
        )

        self.train_btn = ctk.CTkButton(
            train_ctl, text="Start Training", width=220, height=38,
            fg_color=ACCENT_GREEN, hover_color="#2e7d32",
            command=self._start_training,
        )
        self.train_btn.grid(row=3, column=0, columnspan=4, pady=10)

        self.progress = ctk.CTkProgressBar(train_outer)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self.progress.set(0)

        self.stats_label = ctk.CTkLabel(
            train_outer, text="Not trained yet.",
            font=ctk.CTkFont(size=12), text_color=LABEL_COLOUR,
        )
        self.stats_label.grid(row=2, column=0, pady=(0, 2))

        self.tabview = ctk.CTkTabview(win, corner_radius=8)
        self.tabview.grid(row=2, column=0, sticky="nsew", padx=14, pady=(6, 14))

        for tab_name in ("Win Rate", "Reward", "Q-Table Heatmap", "Q-Table"):
            self.tabview.add(tab_name)

        def _centered_canvas(parent: Any, fig: Figure) -> FigureCanvasTkAgg:
            wrapper = ctk.CTkFrame(parent, fg_color="transparent")
            wrapper.pack(fill="both", expand=True, padx=6, pady=6)
            canvas = FigureCanvasTkAgg(fig, wrapper)
            canvas.get_tk_widget().pack(fill="both", expand=True, anchor="center")
            return canvas

        self.wr_fig = Figure(figsize=(5, 2.8), dpi=100, facecolor=FIG_FACECOLOR)
        self.wr_ax = self.wr_fig.add_subplot(111)
        self._style_ax(self.wr_ax, "Episode", "Win Rate", "Win Rate (rolling)")
        self.wr_canvas = _centered_canvas(self.tabview.tab("Win Rate"), self.wr_fig)

        self.rw_fig = Figure(figsize=(5, 2.8), dpi=100, facecolor=FIG_FACECOLOR)
        self.rw_ax = self.rw_fig.add_subplot(111)
        self._style_ax(self.rw_ax, "Episode", "Reward ($)", "Reward History")
        self.rw_canvas = _centered_canvas(self.tabview.tab("Reward"), self.rw_fig)

        self.hm_fig = Figure(figsize=(5, 3.2), dpi=100, facecolor=FIG_FACECOLOR)
        self.hm_ax = self.hm_fig.add_subplot(111)
        self.hm_canvas = _centered_canvas(
            self.tabview.tab("Q-Table Heatmap"), self.hm_fig,
        )

        self.qv_fig = Figure(figsize=(5, 3.2), dpi=100, facecolor=FIG_FACECOLOR)
        self.qv_ax = self.qv_fig.add_subplot(111)
        self.qv_canvas = _centered_canvas(self.tabview.tab("Q-Table"), self.qv_fig)

    # ================================================================
    # Card display helpers
    # ================================================================

    def _clear_frame(self, frame: ctk.CTkFrame) -> None:
        for w in frame.winfo_children():
            w.destroy()

    def _show_hero_cards(self) -> None:
        self._clear_frame(self.hero_cards_frame)
        for c in self.env.hero_cards:
            CardWidget(self.hero_cards_frame, c, face_up=True).pack(
                side="left", padx=4, pady=4,
            )

    def _show_opp_cards(self, face_up: bool = False) -> None:
        self._clear_frame(self.opp_cards_frame)
        if face_up and self.env.opponent_cards:
            for c in self.env.opponent_cards:
                CardWidget(self.opp_cards_frame, c, face_up=True).pack(
                    side="left", padx=4, pady=4,
                )
        else:
            for _ in range(2):
                CardWidget(self.opp_cards_frame, None, face_up=False).pack(
                    side="left", padx=4, pady=4,
                )

    def _show_board(self, cards: List[Card], animate_last: bool = False) -> None:
        self._clear_frame(self.board_frame)
        for i, c in enumerate(cards):
            cw = CardWidget(self.board_frame, c, face_up=True)
            cw.pack(side="left", padx=3, pady=4)
            if animate_last and i == len(cards) - 1:
                cw.configure(fg_color="#1b5e20")
                self.root.after(
                    100,
                    lambda w=cw: w.configure(fg_color="#e8e8e8")
                    if w.winfo_exists() else None,
                )
                self.root.after(
                    200,
                    lambda w=cw: w.configure(fg_color=CARD_BG)
                    if w.winfo_exists() else None,
                )

    # ================================================================
    # Game display
    # ================================================================

    def _update_display(self, animate_card: bool = False) -> None:
        if self.current_state is None:
            return
        self.total_pot_label.configure(
            text=f"Total Pot: ${self.current_state.pot}",
        )
        self.pot_label.configure(
            text=f"Hero invested: ${self.current_state.hero_invested}  |  "
                 f"Opp invested: ${self.current_state.opponent_invested}",
        )
        self.hero_stack_label.configure(
            text=f"Stack: ${self.current_state.hero_stack}",
        )
        self.opp_stack_label.configure(
            text=f"Stack: ${self.current_state.opponent_stack}",
        )
        self.street_label.configure(
            text=f"Street: {self.current_state.street.upper()}",
        )
        self._show_board(
            list(self.current_state.community), animate_last=animate_card,
        )
        self._update_thought_process()

    def _update_thought_process(self) -> None:
        if self.current_state is None:
            return
        # Keep the last thought process visible after the hand ends
        if self.env.done:
            return
        qvals = self.agent.get_q_values(self.current_state)
        valid = set(self.env.get_valid_actions())
        self.thought_state_label.configure(
            text=f"State: {self.current_state.state_key}",
        )
        if not qvals:
            for w in self._thought_bar_widgets.values():
                w["bar"].set(0)
                w["val"].configure(text="--", text_color="#546e7a")
                w["badge"].configure(text="")
            return

        # Only consider valid actions for best-action marking
        valid_q = {a: q for a, q in qvals.items() if a in valid}
        best_a = max(valid_q, key=valid_q.get) if valid_q else None  # type: ignore[arg-type]

        all_q = [qvals.get(a, 0.0) for a in self.env.actions if a in valid]
        q_min = min(all_q) if all_q else 0.0
        q_max = max(all_q) if all_q else 0.0
        q_range = q_max - q_min if q_max != q_min else 1.0

        for action in self.env.actions:
            w = self._thought_bar_widgets[action]
            if action not in valid:
                # Grey out invalid actions
                w["bar"].set(0)
                w["val"].configure(text="n/a", text_color="#546e7a")
                w["badge"].configure(text="")
                continue
            q = qvals.get(action, 0.0)
            norm = max(0.02, (q - q_min) / q_range)
            w["bar"].set(norm)
            if q >= 0:
                col = WR_LINE_COLOUR if action == best_a else "#a5d6a7"
            else:
                col = "#ef9a9a"
            w["val"].configure(text=f"{q:+.1f}", text_color=col)
            w["badge"].configure(text="*" if action == best_a else "")

    # ================================================================
    # Result / message banner helpers
    # ================================================================

    def _show_result(self, text: str, text_color: str, bg_color: str) -> None:
        """Show game-over result.  Persists until Next Hand is clicked."""
        self._cancel_pending()
        self.msg_frame.configure(fg_color=bg_color)
        self.msg_label.configure(text=text, text_color=text_color)
        self.deal_btn.grid_forget()
        self.next_hand_btn.grid(row=1, column=0, pady=(0, 10))
        self.new_game_btn.configure(state="disabled")
        self.ai_play_btn.configure(state="disabled")
        self.root.bind("<space>", lambda e: self._next_hand())

    def _show_deal_btn(self, street_name: str) -> None:
        """Show the 'Deal Next Card' button between streets."""
        self.msg_frame.configure(fg_color="#004d40")
        self.msg_label.configure(
            text=f"Street settled -- deal the {street_name}?",
            text_color="#b2dfdb",
        )
        self.next_hand_btn.grid_forget()
        self.deal_btn.grid(row=1, column=0, pady=(0, 10))

    def _hide_deal_btn(self) -> None:
        self.deal_btn.grid_forget()
        self.msg_frame.configure(fg_color="transparent")
        self.msg_label.configure(text="", text_color="#ffcc80")

    def _clear_result(self) -> None:
        """Clear result banner, deal button, and action log."""
        self._cancel_pending()
        self._processing = False
        self.msg_frame.configure(fg_color="transparent")
        self.msg_label.configure(text="", text_color="#ffcc80")
        self.next_hand_btn.grid_forget()
        self.deal_btn.grid_forget()
        self._next_hand_mode = None
        self._action_log = []
        self.log_label.configure(text="No actions yet.", text_color="#546e7a")
        self.new_game_btn.configure(state="normal")
        self.ai_play_btn.configure(state="normal")
        self.root.unbind("<space>")

    def _next_hand(self) -> None:
        mode = self._next_hand_mode
        self._clear_result()
        if mode == "ai":
            self._watch_ai()
        else:
            self._start_manual()

    # ================================================================
    # "Deal Next Card" button handler
    # ================================================================

    def _deal_next_card(self) -> None:
        """Called when the player (or AI timer) clicks 'Deal Next Card'."""
        if not self.game_active or not self.env.street_settled or self._processing:
            return
        self._processing = True
        self._hide_deal_btn()
        self.env.advance_street()
        self.current_state = self.env._get_state()
        self._update_display(animate_card=True)

        # Log the new street
        self._log_action(
            "Dealer", "deal", self.env.street, bet_amount=0, pot=self.env.pot,
        )

        label = "AI" if self._next_hand_mode == "ai" else "You"
        self._processing = False

        # Whoever did NOT act last on the previous street opens this one.
        if self.env.current_player == "opponent":
            self.msg_label.configure(
                text="New card dealt -- Opponent is acting...",
                text_color="#80cbc4",
            )
            self._disable_actions()
            self._schedule(1000, lambda: self._do_opponent_turn(label))
        else:
            # Hero opens the new street
            if self._next_hand_mode == "ai":
                self.msg_label.configure(
                    text="New card dealt -- AI is thinking...",
                    text_color="#80cbc4",
                )
                self._disable_actions()
                self._schedule(1000, self._ai_step)
            else:
                self.msg_label.configure(
                    text="New card dealt -- Your turn.",
                    text_color="#ffcc80",
                )
                self._enable_actions(self.env.get_valid_actions())

    # ================================================================
    # Action log
    # ================================================================

    def _log_action(self, who: str, action: str, street: str,
                    bet_amount: int = 0, pot: int = 0) -> None:
        if action == "deal":
            tag = "Deal"
            delta = ""
        elif action == "call" and bet_amount == 0:
            tag = "Check"
            delta = ""
        elif action == "call":
            tag = "Call"
            delta = f"  (+${bet_amount})"
        elif action == "fold":
            tag = "Fold"
            delta = ""
        elif action == "raise_150":
            tag = "All-In"
            delta = f"  (+${bet_amount})" if bet_amount > 0 else ""
        elif action.startswith("raise_"):
            tag = "Raise"
            delta = f"  (+${bet_amount})" if bet_amount > 0 else ""
        else:
            tag = action
            delta = f"  (+${bet_amount})" if bet_amount > 0 else ""

        line = (f"{street.upper():>8}  |  {who:<10} -> {tag}"
                f"{delta}  |  Pot: ${pot}")
        self._action_log.append(line)
        self.log_label.configure(
            text="\n".join(self._action_log), text_color="#b0bec5",
        )

    # ================================================================
    # Action button management
    # ================================================================

    def _enable_actions(self, valid: List[str]) -> None:
        self.fold_btn.configure(
            state="normal" if "fold" in valid else "disabled",
        )
        self.call_btn.configure(
            state="normal" if "call" in valid else "disabled",
        )
        self.raise50_btn.configure(
            state="normal" if "raise_50" in valid else "disabled",
        )
        self.raise100_btn.configure(
            state="normal" if "raise_100" in valid else "disabled",
        )
        self.raise150_btn.configure(
            state="normal" if "raise_150" in valid else "disabled",
        )
        if self.current_state:
            stack = self.current_state.hero_stack
            self.raise150_btn.configure(text=f"All-In ${stack}")

    def _disable_actions(self) -> None:
        for btn in (self.fold_btn, self.call_btn, self.raise50_btn,
                    self.raise100_btn, self.raise150_btn):
            btn.configure(state="disabled")

    # ================================================================
    # Shared: process one opponent step result
    # ================================================================

    def _handle_opp_result(self, result: StepResult, label: str = "You") -> None:
        """Process a step_opponent() result: log, update, decide next."""
        info = result.info
        action = info.get("action", "?")
        bet = info.get("bet", 0)
        street_settled = info.get("street_settled", False)

        # Log on the street BEFORE any potential settle
        self._log_action(
            "Opponent", action,
            self.current_state.street if self.current_state else "?",
            bet_amount=bet, pot=self.env.pot,
        )

        self.current_state = result.next_state
        self._update_display()

        # Game over?
        if result.done:
            self.game_active = False
            self._show_opp_cards(face_up=True)
            winner = info.get("winner", "?")
            hero_h = info.get("hero_hand", "")
            opp_h = info.get("opponent_hand", "")
            if winner == "hero":
                if action == "fold":
                    self._show_result(
                        f"Opponent folded! {label} win the pot.",
                        text_color="#ffffff", bg_color="#1b5e20",
                    )
                else:
                    self._show_result(
                        f"{label} WON!  {label}: {hero_h}  Opp: {opp_h}",
                        text_color="#ffffff", bg_color="#1b5e20",
                    )
            elif winner == "opponent":
                self._show_result(
                    f"{label} lost.  {label}: {hero_h}  Opp: {opp_h}",
                    text_color="#ffffff", bg_color="#b71c1c",
                )
            else:
                self._show_result(
                    f"Tie!  (odd chip to hero)",
                    text_color="#000000", bg_color="#fff176",
                )
            return

        # Street settled?  Show the "Deal Next Card" button.
        if street_settled:
            next_street = "Turn" if self.env.street == "flop" else "River"
            self._disable_actions()
            self._show_deal_btn(next_street)
            # In AI mode, auto-click after a delay
            if self._next_hand_mode == "ai":
                self._schedule(1200, self._deal_next_card)
            return

        # Opponent acted but street continues -- hero's turn
        if self.env.current_player == "hero":
            if action.startswith("raise"):
                self.msg_label.configure(
                    text="Opponent raised -- your turn to respond.",
                    text_color="#ffcc80",
                )
            else:
                self.msg_label.configure(
                    text="Your turn.", text_color="#ffcc80",
                )
            # In AI mode, let the AI take hero's turn after a delay
            if self._next_hand_mode == "ai":
                self.msg_label.configure(
                    text="AI is thinking...", text_color="#80cbc4",
                )
                self._schedule(1000, self._ai_step)
            else:
                self._enable_actions(self.env.get_valid_actions())

    # ================================================================
    # Shared: fire ONE opponent turn
    # ================================================================

    def _do_opponent_turn(self, label: str = "You") -> None:
        """Call step_opponent() once and process the result."""
        if (not self.game_active or self.env.done
                or self.env.street_settled or self._processing):
            return
        if self.env.current_player != "opponent":
            return
        self._processing = True
        result = self.env.step_opponent()
        self._processing = False
        self._handle_opp_result(result, label)

    # ================================================================
    # Manual game
    # ================================================================

    def _start_manual(self) -> None:
        if self.training_active:
            return
        self._cancel_pending()
        self._clear_result()
        self.game_active = True
        self._next_hand_mode = "manual"
        self.current_state = self.env.reset()
        self._show_hero_cards()
        self._show_opp_cards(face_up=False)
        self._update_display()

        # Opponent opens -- fire after a short delay
        self.msg_label.configure(
            text="Opponent is acting...", text_color="#80cbc4",
        )
        self._disable_actions()
        self._schedule(500, lambda: self._do_opponent_turn("You"))

    def _player_action(self, action: str) -> None:
        """Hero takes ONE action.  Then fire opponent turn if needed."""
        if (not self.game_active or self.env.done
                or self.env.street_settled or self._processing):
            return
        if self.env.current_player != "hero":
            return
        self._processing = True
        self._disable_actions()

        street_before = self.current_state.street if self.current_state else "?"
        result: StepResult = self.env.step(action)

        hero_bet = result.info.get("bet", 0)
        street_settled = result.info.get("street_settled", False)

        self._log_action("You", action, street_before,
                         bet_amount=hero_bet, pot=self.env.pot)

        self.current_state = result.next_state
        self._update_display()

        # Game over?
        if result.done:
            self.game_active = False
            self._processing = False
            winner = result.info.get("winner", "?")
            hero_h = result.info.get("hero_hand", "")
            opp_h = result.info.get("opponent_hand", "")
            self._show_opp_cards(face_up=True)
            if winner == "hero":
                self._show_result(
                    f"You WON!  Reward: ${result.reward:+.0f}"
                    f"  |  You: {hero_h}  Opp: {opp_h}",
                    text_color="#ffffff", bg_color="#1b5e20",
                )
            elif winner == "opponent":
                self._show_result(
                    f"You lost.  Reward: ${result.reward:+.0f}"
                    f"  |  You: {hero_h}  Opp: {opp_h}",
                    text_color="#ffffff", bg_color="#b71c1c",
                )
            else:
                self._show_result(
                    f"Tie!  Reward: ${result.reward:+.0f}  (odd chip to hero)",
                    text_color="#000000", bg_color="#fff176",
                )
            return

        # Street settled?  Show "Deal Next Card".
        if street_settled:
            self._processing = False
            next_street = "Turn" if self.env.street == "flop" else "River"
            self._show_deal_btn(next_street)
            return

        # Hero raised -> opponent responds after a delay
        self._processing = False
        if self.env.current_player == "opponent":
            self.msg_label.configure(
                text="Opponent is thinking...", text_color="#80cbc4",
            )
            self._schedule(1000, lambda: self._do_opponent_turn("You"))
        else:
            self._enable_actions(self.env.get_valid_actions())

    # ================================================================
    # Watch AI Play
    # ================================================================

    def _watch_ai(self) -> None:
        """Start an AI-played hand.  Uses recursive after() for pacing."""
        if self.training_active:
            return
        if not self.agent.episode_rewards:
            self.msg_frame.configure(fg_color="#b71c1c")
            self.msg_label.configure(
                text="Train the AI first!  Open Train & Analyse and run training.",
                text_color="#ffffff",
            )
            return
        self._cancel_pending()
        self._clear_result()
        self.game_active = True
        self._next_hand_mode = "ai"
        self.current_state = self.env.reset()
        self._show_hero_cards()
        self._show_opp_cards(face_up=False)
        self._update_display()
        self._disable_actions()

        # Opponent opens
        self.msg_label.configure(
            text="Opponent is acting...", text_color="#80cbc4",
        )
        self._schedule(500, lambda: self._do_opponent_turn("AI"))

    def _ai_step(self) -> None:
        """AI (hero) takes ONE action, then delegates to opponent/deal."""
        if (not self.game_active or self.env.done
                or self.env.street_settled or self._processing):
            return
        if self.env.current_player != "hero":
            return
        self._processing = True

        valid = self.env.get_valid_actions()
        if not valid:
            self._processing = False
            return
        action = self.agent.get_action(self.current_state, valid, training=False)

        street_before = self.current_state.street if self.current_state else "?"
        result = self.env.step(action)

        hero_bet = result.info.get("bet", 0)
        street_settled = result.info.get("street_settled", False)

        self._log_action("AI", action, street_before,
                         bet_amount=hero_bet, pot=self.env.pot)
        self.msg_label.configure(
            text=f"AI chose: {action.upper()}", text_color="#80cbc4",
        )

        self.current_state = result.next_state
        self._update_display()

        # Game over?
        if result.done:
            self.game_active = False
            self._processing = False
            winner = result.info.get("winner", "?")
            hero_h = result.info.get("hero_hand", "")
            opp_h = result.info.get("opponent_hand", "")
            self._show_opp_cards(face_up=True)
            if winner == "hero":
                self._show_result(
                    f"AI WON!  Reward: ${result.reward:+.0f}"
                    f"  |  AI: {hero_h}  Opp: {opp_h}",
                    text_color="#ffffff", bg_color="#1b5e20",
                )
            elif winner == "opponent":
                self._show_result(
                    f"AI lost.  Reward: ${result.reward:+.0f}"
                    f"  |  AI: {hero_h}  Opp: {opp_h}",
                    text_color="#ffffff", bg_color="#b71c1c",
                )
            else:
                self._show_result(
                    f"Tie!  Reward: ${result.reward:+.0f}  (odd chip to hero)",
                    text_color="#000000", bg_color="#fff176",
                )
            return

        # Street settled?  Auto-deal after a delay.
        if street_settled:
            self._processing = False
            next_street = "Turn" if self.env.street == "flop" else "River"
            self._show_deal_btn(next_street)
            self._schedule(1200, self._deal_next_card)
            return

        # AI raised -> opponent responds after a delay
        self._processing = False
        if self.env.current_player == "opponent":
            self.msg_label.configure(
                text="Opponent is thinking...", text_color="#80cbc4",
            )
            self._schedule(1000, lambda: self._do_opponent_turn("AI"))

    # ================================================================
    # Training
    # ================================================================

    def _start_training(self) -> None:
        if self.training_active:
            return
        try:
            n = int(self.ep_var.get())
            lr = float(self.lr_var.get())
            gamma = float(self.gamma_var.get())
            eps = float(self.eps_var.get())
            assert n > 0
        except (ValueError, AssertionError):
            self.stats_label.configure(text="Invalid parameters!")
            return

        self.agent = QLearningAgent(
            actions=self.env.actions,
            learning_rate=lr, discount_factor=gamma, epsilon=eps,
        )
        self.training_active = True
        self.train_btn.configure(state="disabled", text="Training...")
        self.progress.set(0)

        thread = threading.Thread(
            target=self._train_worker, args=(n,), daemon=True,
        )
        thread.start()

    def _train_worker(self, n: int) -> None:
        update_interval = max(1, n // 100)
        env = PokerEnv()
        for ep in range(n):
            self.agent.train_episode(env)
            if (ep + 1) % update_interval == 0 or ep == n - 1:
                frac = (ep + 1) / n
                self.root.after(0, self._training_progress, frac)
        self.root.after(0, self._training_done)

    def _training_progress(self, frac: float) -> None:
        self.progress.set(frac)
        stats = self.agent.get_statistics()
        self.stats_label.configure(
            text=(
                f"Episodes: {stats['total_episodes']}  |  "
                f"Win Rate: {stats['win_rate']:.1%}  |  "
                f"Avg Reward: ${stats['avg_reward']:.1f}"
            ),
        )

    def _training_done(self) -> None:
        self.training_active = False
        self.train_btn.configure(state="normal", text="Start Training")
        self.progress.set(1)

        self._plot_winrate()
        self._plot_reward()
        self._plot_heatmap()
        self._update_qv_text()
        self._update_thought_process()

        stats = self.agent.get_statistics()
        self.stats_label.configure(
            text=(
                f"Done!  {stats['total_episodes']} eps  |  "
                f"Win Rate: {stats['win_rate']:.1%}  |  "
                f"Avg Reward: ${stats['avg_reward']:.1f}"
            ),
            text_color=WR_LINE_COLOUR,
        )

    # ================================================================
    # Plotting helpers
    # ================================================================

    @staticmethod
    def _style_ax(
        ax: matplotlib.axes.Axes,
        xlabel: str, ylabel: str, title: str,
    ) -> None:
        ax.set_facecolor(AX_FACECOLOR)
        ax.set_xlabel(xlabel, color=LABEL_COLOUR, fontsize=9)
        ax.set_ylabel(ylabel, color=LABEL_COLOUR, fontsize=9)
        ax.set_title(
            title, color=TITLE_COLOUR, fontsize=13,
            fontweight="bold", pad=12,
        )
        ax.tick_params(colors=TICK_COLOUR, labelsize=8)
        ax.grid(axis="both", color=GRID_COLOUR, linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOUR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _plot_winrate(self) -> None:
        stats = self.agent.get_statistics()
        wins = stats["all_wins"]
        if len(wins) < 2:
            return
        self.wr_ax.clear()
        self._style_ax(
            self.wr_ax, "Episode", "Win Rate", "Win Rate (50-ep rolling)",
        )
        w = min(50, len(wins))
        ma = np.convolve(wins, np.ones(w) / w, mode="valid")
        x = np.arange(w, len(wins) + 1)
        self.wr_ax.fill_between(x, ma, alpha=WR_FILL_ALPHA, color=WR_LINE_COLOUR)
        self.wr_ax.plot(x, ma, color=WR_LINE_COLOUR, linewidth=2)
        self.wr_ax.axhline(
            0.5, color=BASELINE_COLOUR, ls="--", lw=1, label="50% baseline",
        )
        self.wr_ax.set_ylim(0, 1)
        self.wr_ax.legend(
            fontsize=8, facecolor=LEGEND_BG, edgecolor=LEGEND_EDGE,
            labelcolor=LEGEND_TEXT, loc="lower right",
        )
        self.wr_fig.tight_layout()
        self.wr_canvas.draw()

    def _plot_reward(self) -> None:
        stats = self.agent.get_statistics()
        rews = stats["all_rewards"]
        if len(rews) < 2:
            return
        self.rw_ax.clear()
        self._style_ax(self.rw_ax, "Episode", "Reward ($)", "Reward History")
        self.rw_ax.plot(rews, color=RW_RAW_COLOUR, alpha=0.18, linewidth=0.5)
        w = min(50, len(rews))
        ma = np.convolve(rews, np.ones(w) / w, mode="valid")
        x = np.arange(w, len(rews) + 1)
        self.rw_ax.fill_between(x, ma, alpha=RW_FILL_ALPHA, color=RW_MA_COLOUR)
        self.rw_ax.plot(
            x, ma, color=RW_MA_COLOUR, linewidth=2,
            label=f"{w}-ep moving avg",
        )
        self.rw_ax.axhline(0, color=BASELINE_COLOUR, ls="-", lw=0.8)
        self.rw_ax.legend(
            fontsize=8, facecolor=LEGEND_BG, edgecolor=LEGEND_EDGE,
            labelcolor=LEGEND_TEXT, loc="lower right",
        )
        self.rw_fig.tight_layout()
        self.rw_canvas.draw()

    def _plot_heatmap(self) -> None:
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        import matplotlib.patheffects as pe

        snap = self.agent.get_q_table_snapshot()
        if not snap:
            return

        actions = self.env.actions
        action_labels = ["Fold", "Call", "Raise $50", "Raise $100", "All-In"]
        states = sorted(snap.keys())

        def _pretty_state(sk: str) -> str:
            parts = sk.split("_")
            if len(parts) >= 3:
                return f"{parts[0].capitalize()} pot${parts[1]} stk${parts[2]}"
            return sk

        state_labels = [_pretty_state(s) for s in states]
        data = np.array(
            [[snap[s].get(a, 0.0) for a in actions] for s in states],
        )

        cmap = LinearSegmentedColormap.from_list("poker_wr", CMAP_STOPS, N=256)

        self.hm_fig.clear()
        self.hm_ax = self.hm_fig.add_subplot(111)
        self._style_ax(self.hm_ax, "Action", "State", "Q-Table Heatmap")

        vmin, vmax = data.min(), data.max()
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
        norm = Normalize(vmin=vmin, vmax=vmax)

        im = self.hm_ax.imshow(
            data, aspect="auto", cmap=cmap, interpolation="bilinear",
            norm=norm,
        )
        self.hm_ax.set_xticks(np.arange(len(actions)) - 0.5, minor=True)
        self.hm_ax.set_yticks(np.arange(len(states)) - 0.5, minor=True)
        self.hm_ax.grid(
            which="minor", color=GRID_COLOUR, linewidth=0.5, alpha=0.5,
        )
        self.hm_ax.grid(which="major", visible=False)
        self.hm_ax.tick_params(which="minor", length=0)
        self.hm_ax.set_xticks(range(len(actions)))
        self.hm_ax.set_xticklabels(action_labels, fontsize=8, color=TICK_COLOUR)
        self.hm_ax.set_yticks(range(len(states)))
        self.hm_ax.set_yticklabels(state_labels, fontsize=8, color=TICK_COLOUR)

        for i in range(len(states)):
            row_best = int(np.argmax(data[i]))
            for j in range(len(actions)):
                val = data[i, j]
                brightness = norm(val)
                txt_col = FIG_FACECOLOR if brightness > 0.7 else TITLE_COLOUR
                label = f"${val:+.0f}"
                if j == row_best and val != 0:
                    label = f"* ${val:+.0f}"
                txt = self.hm_ax.text(
                    j, i, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_col,
                )
                txt.set_path_effects([
                    pe.withStroke(
                        linewidth=2, foreground=f"{FIG_FACECOLOR}99",
                    ),
                ])

        cbar = self.hm_fig.colorbar(
            im, ax=self.hm_ax, fraction=0.046, pad=0.04,
        )
        cbar.ax.tick_params(colors=TICK_COLOUR, labelsize=8)
        cbar.set_label("Q-Value ($)", color=LABEL_COLOUR, fontsize=9)
        cbar.outline.set_edgecolor(GRID_COLOUR)

        self.hm_fig.tight_layout()
        self.hm_canvas.draw()

    def _update_qv_text(self) -> None:
        import matplotlib.patheffects as pe
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        snap = self.agent.get_q_table_snapshot()
        if not snap:
            return

        actions = self.env.actions
        col_headers = ["Fold", "Call", "Raise $50", "Raise $100", "All-In"]
        states = sorted(snap.keys())

        def _pretty(sk: str) -> str:
            parts = sk.split("_")
            if len(parts) >= 3:
                return f"{parts[0].capitalize()} pot${parts[1]} stk${parts[2]}"
            return sk

        row_labels = [_pretty(s) for s in states]
        data = np.array(
            [[snap[s].get(a, 0.0) for a in actions] for s in states],
        )
        n_rows, n_cols = data.shape

        cmap = LinearSegmentedColormap.from_list("qtab_wr", CMAP_STOPS, N=256)
        vmin, vmax = data.min(), data.max()
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
        norm = Normalize(vmin=vmin, vmax=vmax)

        self.qv_fig.clear()
        self.qv_ax = self.qv_fig.add_subplot(111)
        self.qv_ax.set_facecolor(AX_FACECOLOR)
        self.qv_ax.axis("off")

        cell_text = []
        for i in range(n_rows):
            row = []
            best_j = int(np.argmax(data[i]))
            for j in range(n_cols):
                v = data[i, j]
                s = f"${v:+.0f}"
                if j == best_j and v != 0:
                    s = f"* {s}"
                row.append(s)
            cell_text.append(row)

        tbl = self.qv_ax.table(
            cellText=cell_text, rowLabels=row_labels,
            colLabels=col_headers, cellLoc="center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.8)

        for (row_idx, col_idx), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID_COLOUR)
            cell.set_linewidth(0.5)
            if row_idx == 0:
                cell.set_facecolor(LEGEND_BG)
                cell.set_text_props(
                    color=TITLE_COLOUR, fontweight="bold", fontsize=9,
                )
            elif col_idx == -1:
                cell.set_facecolor(AX_FACECOLOR)
                cell.set_text_props(
                    color=LABEL_COLOUR, fontweight="bold", fontsize=8,
                )
            else:
                q = data[row_idx - 1, col_idx]
                brightness = norm(q)
                cell.set_facecolor(cmap(brightness))
                txt_col = FIG_FACECOLOR if brightness > 0.7 else TITLE_COLOUR
                cell.set_text_props(
                    color=txt_col, fontweight="bold", fontsize=9,
                )
                cell.get_text().set_path_effects([
                    pe.withStroke(
                        linewidth=1.5, foreground=f"{FIG_FACECOLOR}99",
                    ),
                ])

        self.qv_ax.set_title(
            "Q-Table   Values by State x Action",
            color=TITLE_COLOUR, fontsize=13, fontweight="bold", pad=14,
        )
        self.qv_fig.tight_layout()
        self.qv_canvas.draw()

    # ================================================================
    # Run
    # ================================================================

    def run(self) -> None:
        self.root.mainloop()
