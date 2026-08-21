from __future__ import annotations


class GridWorld:
    """
    A simple 4x4 grid world environment for Reinforcement Learning.
    Start: (0, 0)
    Goal: (3, 3)
    Actions: 0 (Up), 1 (Right), 2 (Down), 3 (Left)
    """

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.state: tuple[int, int] = (0, 0)
        self.goal: tuple[int, int] = (size - 1, size - 1)
        self.action_space: list[int] = [0, 1, 2, 3]
        self.observation_space: int = size * size

    def reset(self) -> int:
        self.state = (0, 0)
        return self._get_state_index(self.state)

    def _get_state_index(self, state: tuple[int, int]) -> int:
        return state[0] * self.size + state[1]

    def step(self, action: int) -> tuple[int, float, bool]:
        row, col = self.state

        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)

        self.state = (row, col)

        done = self.state == self.goal
        reward = 1.0 if done else -0.01

        return self._get_state_index(self.state), reward, done
