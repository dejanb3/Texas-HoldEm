"""
Q-Learning Agent for Simplified Texas Hold'em
==============================================
Implements the **Controller** layer (MVC) — the ε-greedy Q-Learning agent
that learns an approximate Nash-Equilibrium policy for the simplified
post-flop heads-up sub-game.

Algorithm
---------
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

Hyperparameters
---------------
* ``learning_rate``  (α)  — step size for updates.
* ``discount_factor`` (γ) — how much future rewards are valued.
* ``epsilon``         (ε) — exploration probability in ε-greedy.

Convergence insight
-------------------
The hero hand 8♥ 9♥ on a J♥ Q♥ 2♣ flop is a *double-gutter* straight draw
(needs T or K → 8 outs) **plus** a flush draw (9 heart outs).  Combined
unique outs ≈ 15, giving ~54% equity to improve by the river.  The agent
should converge to heavily favouring *call* / *raise* on the flop & turn.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from config import (
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_EPSILON,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VERBOSE_EVERY,
)
from environment import GameState, PokerEnv, StepResult


# ============================================================================
# Q-Table type aliases
# ============================================================================

QTable = DefaultDict[str, DefaultDict[str, float]]


def _make_qtable() -> QTable:
    """Factory for a nested default-dict Q-table."""
    return defaultdict(lambda: defaultdict(float))


# ============================================================================
# Q-Learning Agent
# ============================================================================

class QLearningAgent:
    """ε-greedy Q-Learning agent for the simplified poker MDP.

    Parameters
    ----------
    actions : list[str]
        Full action vocabulary (e.g. ``['fold','call','raise_100','raise_150']``).
    learning_rate : float
        α — controls how fast Q-values are updated (0 < α ≤ 1).
    discount_factor : float
        γ — weight of future rewards (0 ≤ γ ≤ 1).
    epsilon : float
        ε — probability of choosing a *random* action (exploration).
    """

    def __init__(
        self,
        actions: List[str],
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.actions: List[str] = list(actions)
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.q_table: QTable = _make_qtable()

        # Training history
        self.episode_rewards: List[float] = []
        self.episode_wins: List[int] = []

    # -- State abstraction ---------------------------------------------------

    @staticmethod
    def _state_key(state: GameState) -> str:
        """Map the full game state to a compact Q-table key.

        The key is ``"<street>_<pot>_<hero_stack>"`` which gives a manageable
        state space while still capturing the essential decision variables.
        """
        return state.state_key

    # -- Action selection ----------------------------------------------------

    def get_action(
        self,
        state: GameState,
        valid_actions: List[str],
        training: bool = True,
    ) -> str:
        """Select an action using the ε-greedy policy.

        During evaluation (``training=False``) the agent is purely greedy.
        """
        if not valid_actions:
            raise ValueError("No valid actions available")

        key = self._state_key(state)

        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitation — break ties randomly
        q_vals = {a: self.q_table[key][a] for a in valid_actions}
        max_q = max(q_vals.values())
        best = [a for a, q in q_vals.items() if q == max_q]
        return random.choice(best)

    # -- Learning ------------------------------------------------------------

    def update(
        self,
        state: GameState,
        action: str,
        reward: float,
        next_state: GameState,
        done: bool,
        valid_next_actions: List[str],
    ) -> None:
        """Apply one-step Q-learning update.

        .. math::

           Q(s,a) <- Q(s,a) + alpha [r + gamma * max_{a'} Q(s',a') - Q(s,a)]
        """
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        current_q: float = self.q_table[key][action]

        if done or not valid_next_actions:
            max_next_q: float = 0.0
        else:
            max_next_q = max(self.q_table[next_key][a] for a in valid_next_actions)

        td_target = reward + self.discount_factor * max_next_q
        self.q_table[key][action] = current_q + self.learning_rate * (td_target - current_q)

    # -- Episode runners -----------------------------------------------------

    def train_episode(self, env: PokerEnv, verbose: bool = False) -> Tuple[float, bool]:
        """Run one training episode using backward pass (Return G)."""
        state = env.reset()
        done = False
        trajectory = [] # Lista za čuvanje (stanje, akcija) parova

       
        while not done:
            if env.street_settled:
                env.advance_street()
                state = env._get_state()
                continue

            if env.current_player == "opponent":
                result = env.step_opponent()
                done, info, state = result.done, result.info, result.next_state
                final_reward = result.reward 
            else:
                valid = env.get_valid_actions()
                action = self.get_action(state, valid, training=True)
                
                next_state, reward, done, info = env.step(action)
                
                
                trajectory.append((state, action))
                
                if not done and env.street_settled:
                    env.advance_street()
                    next_state = env._get_state()
                
                state = next_state
                final_reward = reward # Profit na kraju

        
        g = final_reward 
        for s_t, a_t in reversed(trajectory):
            
            
            self.update(s_t, a_t, g, s_t, True, []) 
            
            # Ponderisanje gamom 
            g *= self.discount_factor

        won = info.get("winner") == "hero"
        self.episode_rewards.append(final_reward)
        self.episode_wins.append(1 if won else 0)
        return final_reward, won

    def train(
        self,
        env: PokerEnv,
        num_episodes: int = 1000,
        verbose_every: int = DEFAULT_VERBOSE_EVERY,
        callback: Optional[Any] = None,
    ) -> None:
        """Train for *num_episodes* episodes.

        Parameters
        ----------
        callback : callable, optional
            Called as ``callback(episode, total_reward, won)`` after each
            episode — used by the GUI for live updates.
        """
        for ep in range(num_episodes):
            reward, won = self.train_episode(env)

            if callback is not None:
                callback(ep, reward, won)

            if verbose_every and (ep + 1) % verbose_every == 0:
                recent_r = self.episode_rewards[-verbose_every:]
                recent_w = self.episode_wins[-verbose_every:]
                avg_r = float(np.mean(recent_r))
                wr = float(np.mean(recent_w))
                print(
                    f"Episode {ep + 1:>6}/{num_episodes}  |  "
                    f"Avg Reward: ${avg_r:>7.2f}  |  "
                    f"Win Rate: {wr:>6.1%}"
                )

    # -- Inspection ----------------------------------------------------------

    def get_q_values(self, state: GameState) -> Dict[str, float]:
        """Return Q-values for every action in *state*."""
        key = self._state_key(state)
        return dict(self.q_table[key])

    def get_q_table_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a plain-dict deep copy of the entire Q-table.

        Useful for serialisation and for the GUI heatmap.
        """
        return {
            state_key: dict(action_vals)
            for state_key, action_vals in self.q_table.items()
        }

    def get_statistics(self, window: int = 100) -> Dict[str, Any]:
        """Compute rolling statistics over training history."""
        if not self.episode_rewards:
            return {
                "avg_reward": 0.0,
                "win_rate": 0.0,
                "total_episodes": 0,
                "all_rewards": [],
                "all_wins": [],
            }
        recent_r = self.episode_rewards[-window:]
        recent_w = self.episode_wins[-window:]
        return {
            "avg_reward": float(np.mean(recent_r)),
            "win_rate": float(np.mean(recent_w)),
            "total_episodes": len(self.episode_rewards),
            "all_rewards": list(self.episode_rewards),
            "all_wins": list(self.episode_wins),
        }

    # -- Persistence ---------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Persist the Q-table and training history to *filepath*."""
        data = {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "episode_rewards": self.episode_rewards,
            "episode_wins": self.episode_wins,
            "hyperparams": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
            },
        }
        with open(filepath, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Agent saved → {filepath}")

    def load(self, filepath: str) -> None:
        """Load a previously saved agent from *filepath*."""
        with open(filepath, "rb") as fh:
            data = pickle.load(fh)
        self.q_table = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in data["q_table"].items()},
        )
        self.episode_rewards = data.get("episode_rewards", [])
        self.episode_wins = data.get("episode_wins", [])
        hp = data.get("hyperparams", {})
        if hp:
            self.learning_rate = hp.get("learning_rate", self.learning_rate)
            self.discount_factor = hp.get("discount_factor", self.discount_factor)
            self.epsilon = hp.get("epsilon", self.epsilon)
        print(f"Agent loaded ← {filepath}")
