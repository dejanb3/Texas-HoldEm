"""
Poker Environment -- Simplified Heads-Up Post-Flop MDP
======================================================
Implements the **Model** layer for a simplified heads-up Texas Hold'em
starting from the Flop with fixed hero cards.

Key design decisions
--------------------
* **Hand ranking engine** -- robust 5-of-7 evaluator following the strict
  BGC hierarchy: Royal Flush > Straight Flush > 4-of-a-Kind > Full House >
  Flush > Straight > 3-of-a-Kind > 2 Pair > 1 Pair > High Card.
* **Ace dynamics** -- Ace is high (A K Q J T) *and* low (5 4 3 2 A).
* **Pot splitting** -- on tied ranks the pot is split; odd chips go to
  hero (player left of the dealer, as per BGC rules).
* **Opponent** -- treated as a *fixed, stochastic* part of the
  environment (analogous to the Dealer in Blackjack).
* **State space** -- ``(street, hero_stack, community_cards_tuple)``.
* **Reward** -- ``R = final_stack - INITIAL_STACK``.

Atomic-Action + Manual Street Advance
--------------------------------------
Each call to ``step()`` / ``step_opponent()`` executes **exactly one**
player action and returns.

A street settles when **both** players have acted **and** their
per-street bets are equal (or a player is all-in).  This rule
applies uniformly on the Flop, Turn, and River.

* **Flop / Turn:** when bets match the engine sets
  ``street_settled = True`` but does **NOT** deal the next card.
  The caller must explicitly call ``advance_street()`` to deal
  the Turn or River.  This makes the UI responsible for pacing.
* **River:** when bets match the engine goes straight to showdown.

Public API
----------
* :class:`Card`, :class:`HandRank`, :class:`HandEvaluator`
* :class:`OpponentPolicy`, :class:`GameState`, :class:`StepResult`
* :class:`PokerEnv`
    - ``reset()``
    - ``step(action)``       -- hero acts once
    - ``step_opponent()``    -- opponent acts once
    - ``advance_street()``   -- deal next card (Turn / River)
    - ``get_valid_actions()``
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from typing import (
    Any,
    Dict,
    Final,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from config import (
    ACTIONS,
    HAND_NAMES,
    INITIAL_POT,
    INITIAL_STACK,
    OPPONENT_AGGRESSION,
    OPPONENT_FOLD_PROB,
    RANKS,
    STOCHASTIC,
    SUIT_SYMBOLS,
    SUITS,
)


# ============================================================================
# Card
# ============================================================================

class Card:
    """Immutable representation of a standard playing card."""

    RANKS: Final[List[str]] = RANKS
    SUITS: Final[List[str]] = SUITS
    SUIT_SYMBOLS: Final[Dict[str, str]] = SUIT_SYMBOLS

    __slots__ = ("rank", "suit", "rank_value")

    def __init__(self, rank: str, suit: str) -> None:
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank!r}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit!r}")
        self.rank: str = rank
        self.suit: str = suit
        self.rank_value: int = self.RANKS.index(rank)

    @property
    def symbol(self) -> str:
        return f"{self.rank}{self.SUIT_SYMBOLS[self.suit]}"

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank!r}, {self.suit!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


# ============================================================================
# Hand Evaluator
# ============================================================================

class HandRank(NamedTuple):
    """Comparable hand evaluation result."""
    rank: int
    tiebreakers: Tuple[int, ...]
    name: str


class HandEvaluator:
    """Evaluate poker hands strictly following BGC Texas Hold'em rules.

    Hierarchy (high to low)::

        10  Royal Flush
         9  Straight Flush
         8  Four of a Kind
         7  Full House
         6  Flush
         5  Straight
         4  Three of a Kind
         3  Two Pair
         2  One Pair
         1  High Card

    Ace dynamics:
        * High straight: A K Q J T
        * Low  straight: 5 4 3 2 A  (the *wheel*)
    """

    HAND_NAMES: Final[Dict[int, str]] = HAND_NAMES

    @staticmethod
    def evaluate_hand(cards: Sequence[Card]) -> HandRank:
        if len(cards) < 5: # ovde bi trebalo umesto 5 da bude 9 ?
            raise ValueError(f"Need >= 5 cards, got {len(cards)}") 
        best: Optional[HandRank] = None
        for combo in itertools.combinations(cards, 5):
            hr = HandEvaluator._evaluate_five(list(combo))
            if best is None or hr > best:
                best = hr
        assert best is not None
        return best

    @staticmethod
    def compare_hands(
        hand1_cards: Sequence[Card],
        hand2_cards: Sequence[Card],
    ) -> int:
        hr1 = HandEvaluator.evaluate_hand(hand1_cards)
        hr2 = HandEvaluator.evaluate_hand(hand2_cards)
        if hr1 > hr2:
            return 1
        if hr1 < hr2:
            return -1
        return 0

    @staticmethod
    def hand_name(rank: int) -> str:
        return HandEvaluator.HAND_NAMES.get(rank, "Unknown")

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _evaluate_five(cards: List[Card]) -> HandRank:
        is_flush = HandEvaluator._is_flush(cards)
        is_straight, straight_high = HandEvaluator._is_straight(cards)
        rank_counts = Counter(c.rank_value for c in cards)
        counts_sorted = sorted(rank_counts.values(), reverse=True)

        sorted_groups = sorted(
            rank_counts.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        tiebreakers = tuple(rv for rv, _ in sorted_groups)

        if is_flush and is_straight and straight_high == 12:
            return HandRank(10, (12,), "Royal Flush")
        if is_flush and is_straight:
            return HandRank(9, (straight_high,), "Straight Flush")
        if counts_sorted == [4, 1]:
            return HandRank(8, tiebreakers, "Four of a Kind")
        if counts_sorted == [3, 2]:
            return HandRank(7, tiebreakers, "Full House")
        if is_flush:
            desc = tuple(sorted((c.rank_value for c in cards), reverse=True))
            return HandRank(6, desc, "Flush")
        if is_straight:
            return HandRank(5, (straight_high,), "Straight")
        if counts_sorted == [3, 1, 1]:
            return HandRank(4, tiebreakers, "Three of a Kind")
        if counts_sorted == [2, 2, 1]:
            return HandRank(3, tiebreakers, "Two Pair")
        if counts_sorted == [2, 1, 1, 1]:
            return HandRank(2, tiebreakers, "One Pair")

        desc = tuple(sorted((c.rank_value for c in cards), reverse=True))
        return HandRank(1, desc, "High Card")

    @staticmethod
    def _is_flush(cards: List[Card]) -> bool:
        return len({c.suit for c in cards}) == 1

    @staticmethod
    def _is_straight(cards: List[Card]) -> Tuple[bool, int]:
        ranks = sorted(c.rank_value for c in cards)
        if ranks == list(range(ranks[0], ranks[0] + 5)):
            return True, ranks[-1]
        if ranks == [0, 1, 2, 3, 12]:
            return True, 3
        return False, 0

    # -- Outs helper ---------------------------------------------------------

    @staticmethod
    def count_outs(
        hero_cards: List[Card],
        community: List[Card],
        remaining_deck: List[Card],
    ) -> Dict[str, int]:
        current_pool = hero_cards + community
        current_hr = (
            HandEvaluator.evaluate_hand(current_pool)
            if len(current_pool) >= 5
            else None
        )
        flush_outs = 0
        straight_outs = 0
        improving_cards: List[Card] = []

        for card in remaining_deck:
            test_pool = current_pool + [card]
            if len(test_pool) < 5:
                continue
            test_hr = HandEvaluator.evaluate_hand(test_pool)
            if current_hr is None or test_hr > current_hr:
                improving_cards.append(card)
                if test_hr.name in ("Flush", "Royal Flush", "Straight Flush"):
                    flush_outs += 1
                if test_hr.name in ("Straight", "Straight Flush"):
                    straight_outs += 1

        return {
            "flush": flush_outs,
            "straight": straight_outs,
            "total_unique": len(set(improving_cards)),
        }


# ============================================================================
# Opponent Policy
# ============================================================================

class OpponentPolicy:
    """Fixed stochastic opponent -- part of the environment."""

    def __init__(
        self,
        aggression: float = OPPONENT_AGGRESSION,
        fold_prob: float = OPPONENT_FOLD_PROB,
    ) -> None:
        self.aggression = aggression
        self.fold_prob = fold_prob

    def get_action(
        self,
        state: Dict[str, Any],
        valid_actions: List[str],
    ) -> str:
        if "fold" in valid_actions and random.random() < self.fold_prob:
            return "fold"
        if "raise_100" in valid_actions and random.random() < self.aggression:
            return "raise_100"
        if "call" in valid_actions:
            return "call"
        return valid_actions[0]


# ============================================================================
# State representation
# ============================================================================

class GameState(NamedTuple):
    """Observable game state exposed to the agent."""
    street: str
    hero_cards: Tuple[Card, Card]
    community: Tuple[Card, ...]
    pot: int
    hero_stack: int
    opponent_stack: int
    hero_invested: int
    opponent_invested: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "street": self.street,
            "hero_cards": list(self.hero_cards),
            "community": list(self.community),
            "pot": self.pot,
            "hero_stack": self.hero_stack,
            "opponent_stack": self.opponent_stack,
            "hero_invested": self.hero_invested,
            "opponent_invested": self.opponent_invested,
        }

    @property
    def state_key(self) -> str:
        return f"{self.street}_{self.pot}_{self.hero_stack}"


# ============================================================================
# Step result
# ============================================================================

class StepResult(NamedTuple):
    """Value returned by step() / step_opponent()."""
    next_state: GameState
    reward: float
    done: bool
    info: Dict[str, Any]


# ============================================================================
# Poker Environment  --  Atomic-Action + Manual Street Advance
# ============================================================================

class PokerEnv:
    """Simplified Heads-Up Texas Hold'em -- post-flop MDP.

    Contract
    --------
    * ``step(action)``     -- hero takes **exactly one** action, returns.
    * ``step_opponent()``  -- opponent takes **exactly one** action, returns.
    * A street settles when **both** players have acted and bets are
      equal (or a player is all-in).  This rule is uniform across
      Flop, Turn, and River.
    * **Flop / Turn:** when settled, ``info["street_settled"] = True``.
      The caller must invoke ``advance_street()`` to deal Turn / River.
    * **River:** when settled, the engine proceeds directly to showdown.
    """

    ACTIONS: Final[List[str]] = ACTIONS

    def __init__(self, stochastic: bool = STOCHASTIC) -> None:
        self.stochastic = stochastic

        # Default (fixed) scenario -- overwritten each reset when stochastic
        if not stochastic:
            self.hero_cards: List[Card] = [Card("8", "h"), Card("9", "h")]
            self.flop: List[Card] = [
                Card("J", "h"), Card("Q", "h"), Card("2", "c"),
            ]
        else:
            self.hero_cards: List[Card] = []
            self.flop: List[Card] = []

        self.deck: List[Card] = []
        self.turn: Optional[Card] = None
        self.river: Optional[Card] = None
        self.opponent_cards: List[Card] = []

        self.pot: int = INITIAL_POT
        self.hero_stack: int = INITIAL_STACK
        self.opponent_stack: int = INITIAL_STACK
        self.hero_invested: int = 0
        self.opponent_invested: int = 0

        self.hero_street_bet: int = 0
        self.opp_street_bet: int = 0

        # Per-street action flags -- both must act before a street can settle
        self.hero_acted: bool = False
        self.opp_acted: bool = False

        self.street: str = "flop"
        self.done: bool = False
        self.winner: Optional[str] = None

        # Turn toggle
        self.current_player: str = "opponent"

        # Tracks who acted last -- used by advance_street to alternate openers
        self.last_actor: str = "hero"

        # Street-settled flag -- set when bets match, cleared by advance_street
        self.street_settled: bool = False

        self.opponent_policy = OpponentPolicy(aggression=OPPONENT_AGGRESSION)
        self.actions: List[str] = list(self.ACTIONS)

    # -- chip helpers --------------------------------------------------------

    def _hero_puts(self, amount: int) -> int:
        actual = min(amount, self.hero_stack)
        self.hero_stack -= actual
        self.hero_invested += actual
        self.hero_street_bet += actual
        self.pot += actual
        return actual

    def _opp_puts(self, amount: int) -> int:
        actual = min(amount, self.opponent_stack)
        self.opponent_stack -= actual
        self.opponent_invested += actual
        self.opp_street_bet += actual
        self.pot += actual
        return actual

    # -- public API ----------------------------------------------------------

    def reset(self) -> GameState:
        """Start a new hand.  ``current_player`` begins as ``"opponent"``.

        When ``stochastic`` is enabled the hero cards and flop are
        re-dealt randomly from a fresh 52-card deck each hand.
        """
        full_deck = [Card(r, s) for r in Card.RANKS for s in Card.SUITS]
        random.shuffle(full_deck)

        if self.stochastic:
            # Deal random hero cards and flop from the shuffled deck
            self.hero_cards = [full_deck.pop(), full_deck.pop()]
            self.flop = [full_deck.pop(), full_deck.pop(), full_deck.pop()]
            self.deck = full_deck            # remaining 47 cards
        else:
            # Fixed scenario -- remove known cards from the deck
            self.deck = [
                c for c in full_deck
                if c not in self.hero_cards and c not in self.flop
            ]

        self.opponent_cards = [self.deck.pop(), self.deck.pop()]

        self.turn = None
        self.river = None
        self.pot = INITIAL_POT
        self.hero_stack = INITIAL_STACK
        self.opponent_stack = INITIAL_STACK
        self.hero_invested = 0
        self.opponent_invested = 0
        self.hero_street_bet = 0
        self.opp_street_bet = 0
        self.hero_acted = False
        self.opp_acted = False
        self.street = "flop"
        self.done = False
        self.winner = None
        self.current_player = "opponent"
        self.last_actor = "hero"  # so opponent opens the flop
        self.street_settled = False
        return self._get_state()

    def get_valid_actions(self) -> List[str]:
        """Actions available to hero right now."""
        if self.done or self.street_settled:
            return []
        valid: List[str] = ["fold", "call"]
        if self.hero_stack >= 50:
            valid.append("raise_50")
        if self.hero_stack >= 100:
            valid.append("raise_100")
        if self.hero_stack > 0:
            valid.append("all_in")
        return valid

    def _opp_valid(self) -> List[str]:
        if self.done or self.street_settled:
            return []
        valid: List[str] = ["fold", "call"]
        if self.opponent_stack >= 100:
            valid.append("raise_100")
        return valid

    # ================================================================
    # advance_street  --  Deal Turn or River.  Must be called explicitly.
    # ================================================================

    def advance_street(self) -> bool:
        """Deal the next community card and reset street bets.

        Returns ``True`` if a card was dealt (flop→turn or turn→river).
        Returns ``False`` if already on the river (should not happen in
        normal flow because river-settle goes to showdown).
        """
        if not self.street_settled:
            return False

        self.street_settled = False

        if self.street == "flop":
            self.turn = self.deck.pop()
            self.street = "turn"
        elif self.street == "turn":
            self.river = self.deck.pop()
            self.street = "river"
        else:
            return False

        self.hero_street_bet = 0
        self.opp_street_bet = 0
        self.hero_acted = False
        self.opp_acted = False
        # The player who acted LAST before the settle sits out;
        # the OTHER player opens the new street.
        self.current_player = "hero" if self.last_actor == "opponent" else "opponent"
        return True

    # ================================================================
    # step_opponent  --  opponent takes EXACTLY ONE action, then STOP
    # ================================================================

    def step_opponent(self) -> StepResult:
        """Opponent acts once.  Returns immediately."""
        if self.done:
            return StepResult(self._get_state(), 0.0, True, {"who": "opponent"})

        action = self.opponent_policy.get_action(
            self._get_state().to_dict(), self._opp_valid(),
        )
        bet: int = 0

        # -- fold --
        if action == "fold":
            self.done = True
            self.winner = "hero"
            reward = float((self.hero_stack + self.pot) - INITIAL_STACK)
            self.current_player = "hero"
            return StepResult(self._get_state(), reward, True,
                              {"who": "opponent", "action": "fold", "bet": 0,
                               "winner": "hero"})

        # -- call / check --
        if action == "call":
            to_call = max(0, self.hero_street_bet - self.opp_street_bet)
            bet = self._opp_puts(to_call)

        # -- raise --
        elif action == "raise_100":
            to_call = max(0, self.hero_street_bet - self.opp_street_bet)
            bet = self._opp_puts(to_call + 100)

        self.opp_acted = True
        self.last_actor = "opponent"

        # Street settles when BOTH players have acted AND either:
        #   (a) bets are exactly equal (normal call / check-check), OR
        #   (b) opponent went all-in but could not exceed hero's bet
        #       ("all-in for less" -- effectively a capped call), OR
        #   (c) BOTH stacks are zero -- no more betting is possible.
        bets_matched = (
            self.hero_acted
            and self.opp_acted
            and (
                self.opp_street_bet == self.hero_street_bet
                or (self.opponent_stack == 0
                    and self.opp_street_bet <= self.hero_street_bet)
                or (self.hero_stack == 0 and self.opponent_stack == 0)
            )
        )

        info: Dict[str, Any] = {
            "who": "opponent", "action": action, "bet": bet,
        }

        if bets_matched:
            if self.street == "river":
                return self._showdown(info_extra=info)
            # Street settled -- do NOT advance.  Wait for advance_street().
            self.street_settled = True
            self.current_player = "hero"  # doesn't matter much; street is paused
            info["street_settled"] = True
            return StepResult(self._get_state(), 0.0, False, info)

        # Not settled -> hero's turn
        self.current_player = "hero"
        return StepResult(self._get_state(), 0.0, False, info)

    # ================================================================
    # step  --  hero takes EXACTLY ONE action, then STOP
    # ================================================================

    def step(self, action: str) -> StepResult:
        """Hero acts once. Returns immediately."""
        if self.done:
            return StepResult(self._get_state(), 0.0, True, {})

        # -- fold -- (Terminalno stanje)
        if action == "fold":
            self.done = True
            self.winner = "opponent"
            reward = float(self.hero_stack - INITIAL_STACK) # Gubitak uloga
            return StepResult(self._get_state(), reward, True, 
                              {"who": "hero", "action": "fold", "winner": "opponent"})

        # -- ulozi -- (Intermedijarni koraci, reward je 0.0)
        hero_bet: int = 0
        if action == "call":
            to_call = max(0, self.opp_street_bet - self.hero_street_bet)
            hero_bet = self._hero_puts(to_call)
        elif action == "raise_50":
            to_call = max(0, self.opp_street_bet - self.hero_street_bet)
            hero_bet = self._hero_puts(to_call + 50)
        elif action == "raise_100":
            to_call = max(0, self.opp_street_bet - self.hero_street_bet)
            hero_bet = self._hero_puts(to_call + 100)
        elif action == "all_in":
            hero_bet = self._hero_puts(self.hero_stack)

        self.hero_acted = True
        self.last_actor = "hero"
        info = {"who": "hero", "action": action, "bet": hero_bet}

        # Provera da li se street završava
        bets_matched = (
            self.hero_acted and self.opp_acted and 
            (self.hero_street_bet == self.opp_street_bet or self.hero_stack == 0)
        )

        if bets_matched:
            if self.street == "river":
                return self._showdown(info_extra=info)
            self.street_settled = True
            info["street_settled"] = True
            return StepResult(self._get_state(), 0.0, False, info)

        self.current_player = "opponent"
        return StepResult(self._get_state(), 0.0, False, info)

    # -- helpers -------------------------------------------------------------

    def get_remaining_deck(self) -> List[Card]:
        return list(self.deck)

    def render(self) -> str:
        community = self._community_list()
        board = " ".join(c.symbol for c in community) if community else "---"
        hero = " ".join(c.symbol for c in self.hero_cards)
        return (
            f"Street : {self.street.upper()}\n"
            f"Hero   : {hero}\n"
            f"Board  : {board}\n"
            f"Pot    : ${self.pot}  |  Hero Stack: ${self.hero_stack}"
        )

    def _community_list(self) -> List[Card]:
        community = list(self.flop)
        if self.turn is not None:
            community.append(self.turn)
        if self.river is not None:
            community.append(self.river)
        return community

    def _get_state(self) -> GameState:
        return GameState(
            street=self.street,
            hero_cards=(self.hero_cards[0], self.hero_cards[1]),
            community=tuple(self._community_list()),
            pot=self.pot,
            hero_stack=self.hero_stack,
            opponent_stack=self.opponent_stack,
            hero_invested=self.hero_invested,
            opponent_invested=self.opponent_invested,
        )

    def _showdown(self, info_extra: Optional[Dict[str, Any]] = None) -> StepResult:
        self.street = "showdown"
        self.done = True

        community = self._community_list()
        hero_pool = list(self.hero_cards) + community
        opp_pool = list(self.opponent_cards) + community

        cmp = HandEvaluator.compare_hands(hero_pool, opp_pool)

        if cmp > 0:
            self.winner = "hero"
            print("gotovo_zavrsio_sam_pobednik_agent")
            reward = float((self.hero_stack + self.pot) - INITIAL_STACK)
        elif cmp < 0:
            self.winner = "opponent"
            print("gotovo_zavrsio_sam_pobednik_protivnik")
            reward = float(self.hero_stack - INITIAL_STACK)
        else:
            self.winner = "tie"
            print("gotovo_zavrsio_sam_izjednaceno")
            #hero_share = self.pot / 2.0
            #reward = float((self.hero_stack + hero_share) - INITIAL_STACK)
            reward = float(self.pot / 2.0)

        hero_hr = HandEvaluator.evaluate_hand(hero_pool)
        opp_hr = HandEvaluator.evaluate_hand(opp_pool)

        info: Dict[str, Any] = {
            "winner": self.winner,
            "hero_hand": hero_hr.name,
            "opponent_hand": opp_hr.name,
            "hero_rank": hero_hr.rank,
            "opponent_rank": opp_hr.rank,
        }
        if info_extra:
            info.update(info_extra)
        return StepResult(self._get_state(), reward, True, info)
