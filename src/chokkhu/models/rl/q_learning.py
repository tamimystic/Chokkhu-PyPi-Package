from __future__ import annotations

from typing import Any

import numpy as np

from ..base import ChokkhuModel


class QLearning(ChokkhuModel):
    def __init__(
        self,
        env: Any = None,
        episodes: int = 1000,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        self.env = env
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.random_state = random_state
        self.q_table: np.ndarray | None = None

    def fit(
        self, X: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> QLearning:
        # For RL, X and y are not used; we use self.env
        if self.env is None:
            # Fallback to default built-in GridWorld if no env provided
            from .environments import GridWorld

            self.env = GridWorld()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_states = getattr(self.env, "observation_space", 16)
        n_actions = len(getattr(self.env, "action_space", [0, 1, 2, 3]))

        self.q_table = np.zeros((n_states, n_actions))

        for _ in range(self.episodes):
            state = self.env.reset()
            done = False

            while not done:
                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.choice(n_actions)
                else:
                    action = int(np.argmax(self.q_table[state, :]))

                next_state, reward, done = self.env.step(action)

                best_next_action = np.argmax(self.q_table[next_state, :])
                td_target = (
                    reward
                    + self.discount_factor * self.q_table[next_state, best_next_action]
                )
                td_error = td_target - self.q_table[state, action]

                self.q_table[state, action] += self.learning_rate * float(td_error)
                state = next_state

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.q_table is None:
            raise ValueError("Model is not fitted yet.")

        # For Q-learning, X can be an array of states (integers)
        predictions = []
        for state in X:
            state_val = state.item() if hasattr(state, "item") else state[0]
            state_idx = int(float(state_val))  # type: ignore
            best_action = np.argmax(self.q_table[state_idx, :])
            predictions.append(best_action)

        return np.array(predictions)
