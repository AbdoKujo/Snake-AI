"""Classe GameState représentant l'état complet du jeu."""

from __future__ import annotations
import time
from collections import deque
from typing import Tuple, List, Optional, Set
import numpy as np

from .point import Point
from .direction import Direction
from .snake import Snake
from .food import Food


class GameState:
    """
    Représente l'état complet du jeu à un instant donné.
    C'est le modèle central utilisé par le contrôleur et les agents IA.
    """

    def __init__(self, grid_width: int, grid_height: int, initial_snake_length: int = 3) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.initial_snake_length = initial_snake_length

        start_x = grid_width // 2
        start_y = grid_height // 2

        self.snake = Snake(Point(start_x, start_y), initial_snake_length)
        self.food  = Food(grid_width, grid_height)

        self.score:      int   = 0
        self.moves:      int   = 0
        self.game_over:  bool  = False
        self.won:        bool  = False
        self.start_time: float = time.time()

        self.food.spawn(self.snake.get_body_set())

    def reset(self) -> None:
        """Réinitialise le jeu à son état initial."""
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2

        self.snake.reset(Point(start_x, start_y), self.initial_snake_length)
        self.food.spawn(self.snake.get_body_set())

        self.score      = 0
        self.moves      = 0
        self.game_over  = False
        self.won        = False
        self.start_time = time.time()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, action: Optional[Direction] = None) -> Tuple[bool, float]:
        """
        Met à jour l'état du jeu après un mouvement.

        Reward policy — 4 signals, no hidden variables, stable Q-targets
        ─────────────────────────────────────────────────────────────────
        Death (any)  : -10.0          flat — no wall/body distinction
        Eat food     : +10.0          flat — no length/efficiency scaling
        Approaching  : +0.1/len       self-fading (+0.033 early → +0.001 late)
        Retreating   : -0.2/len       asymmetric — oscillation net always < 0
        Victory      : +100.0

        Hard per-food step cap handled by the trainer (not here).

        Oscillation math (length=3):
          (+0.033 - 0.067) / 2 = -0.017/step  → always a loss

        Returns:
            Tuple (done, reward).
        """
        if self.game_over:
            return True, 0.0

        food_pos   = self.food.position
        old_head_x = self.snake.head.x
        old_head_y = self.snake.head.y

        if action is not None:
            self.snake.change_direction(action)

        self.snake.move()
        self.moves += 1

        gw, gh = self.grid_width, self.grid_height

        # ── Death ────────────────────────────────────────────────────────
        if self.snake.check_collision(gw, gh):
            self.snake.alive = False
            self.game_over   = True
            return True, -10.0

        snake_len = len(self.snake.body)

        # ── Self-fading proximity signal ─────────────────────────────────
        # Dividing by snake_len means the signal naturally becomes negligible
        # as the snake grows — food reward (+10) dominates in late game.
        old_dist = abs(old_head_x - food_pos.x) + abs(old_head_y - food_pos.y)
        new_dist = abs(self.snake.head.x - food_pos.x) + abs(self.snake.head.y - food_pos.y)
        if new_dist < old_dist:
            reward = 0.1 / snake_len    # +0.033 at len=3 → +0.001 at len=100
        else:
            reward = -0.2 / snake_len   # -0.067 at len=3 → -0.002 at len=100

        # ── Food eaten ───────────────────────────────────────────────────
        if self.food.is_at(self.snake.head):
            self.snake.grow()
            self.score += self.food.value
            reward = 10.0

            if not self.food.spawn(self.snake.get_body_set()):
                self.won       = True
                self.game_over = True
                reward         = 100.0

        return False, reward

    # ------------------------------------------------------------------
    # State representation — 3-channel spatial grid (CNN input)
    # ------------------------------------------------------------------

    def get_state_representation(self) -> np.ndarray:
        """
        Build a 3-channel grid for the CNN-DQN.

        Channels (shape: channels × grid_height × grid_width):
          [0] Snake body — gradient 1.0 (head) → ~0.1 (tail), 0 elsewhere.
                           Encodes both the body layout AND movement direction.
          [1] Food       — 1.0 at food cell, 0 elsewhere.
          [2] Free space — 1.0 at walkable cells (no body, in-bounds), 0 elsewhere.

        Returns:
            np.ndarray of shape (channels * grid_height * grid_width,), dtype float32.
            Flattened so it fits the replay buffer's 1-D state arrays.
            The CNN reshapes it back to (C, H, W) inside forward().
        """
        gw, gh = self.grid_width, self.grid_height
        grid   = np.zeros((3, gh, gw), dtype=np.float32)

        # Channel 0 — snake body gradient --------------------------------
        # Guard bounds: after a wall-collision the head may sit outside the grid.
        body  = self.snake.body
        n     = len(body)
        for i, p in enumerate(body):
            if 0 <= p.x < gw and 0 <= p.y < gh:
                # head (i=0) → 1.0 ; tail (i=n-1) → 0.1
                grid[0, p.y, p.x] = 1.0 - (i / n) * 0.9

        # Channel 1 — food -----------------------------------------------
        food_pos = self.food.position
        if food_pos is not None:
            grid[1, food_pos.y, food_pos.x] = 1.0

        # Channel 2 — free space (1 where the snake can move to) ---------
        # Fill all cells as free, then mark occupied body cells as 0.
        # O(snake_length) instead of O(grid_width * grid_height) Python iters.
        grid[2] = 1.0
        for p in body:
            if 0 <= p.x < gw and 0 <= p.y < gh:
                grid[2, p.y, p.x] = 0.0

        return grid.flatten()

    # ------------------------------------------------------------------
    # Fast BFS helpers — use raw (x,y) tuples, no Point objects in hot loop
    # ------------------------------------------------------------------

    def _bfs_count(self, sx: int, sy: int,
                   obstacles: Set[tuple], gw: int, gh: int) -> int:
        """Count reachable empty cells from (sx,sy). Returns 0 if blocked."""
        if not (0 <= sx < gw and 0 <= sy < gh) or (sx, sy) in obstacles:
            return 0
        visited: Set[tuple] = {(sx, sy)}
        queue: deque = deque([(sx, sy)])
        count = 0
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                if 0 <= nx < gw and 0 <= ny < gh:
                    pos = (nx, ny)
                    if pos not in obstacles and pos not in visited:
                        visited.add(pos)
                        queue.append(pos)
                        count += 1
        return count

    def _bfs_reaches(self, sx: int, sy: int, tx: int, ty: int,
                     obstacles: Set[tuple], gw: int, gh: int) -> bool:
        """Return True if (tx,ty) is reachable from (sx,sy) avoiding obstacles."""
        if sx == tx and sy == ty:
            return True
        target = (tx, ty)
        visited: Set[tuple] = {(sx, sy)}
        queue: deque = deque([(sx, sy)])
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                if 0 <= nx < gw and 0 <= ny < gh:
                    pos = (nx, ny)
                    if pos == target:
                        return True
                    if pos not in obstacles and pos not in visited:
                        visited.add(pos)
                        queue.append(pos)
        return False

    # ------------------------------------------------------------------
    # Helpers used by A* and game controller
    # ------------------------------------------------------------------

    def is_valid_move(self, direction: Direction) -> bool:
        dx, dy  = direction.to_vector()
        new_pos = Point(self.snake.head.x + dx, self.snake.head.y + dy)
        return not self.snake.will_collide_at(new_pos, self.grid_width, self.grid_height)

    def get_valid_moves(self) -> List[Direction]:
        return [
            d for d in Direction.all_directions()
            if not d.is_opposite(self.snake.direction) and self.is_valid_move(d)
        ]

    def get_empty_cells(self) -> List[Point]:
        occupied = self.snake.get_body_set()
        return [
            Point(x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if Point(x, y) not in occupied
        ]

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time

    def copy(self) -> GameState:
        """Deep copy for A* simulation."""
        new_state = GameState(self.grid_width, self.grid_height, self.initial_snake_length)

        new_state.snake.body      = [p.copy() for p in self.snake.body]
        new_state.snake.direction = self.snake.direction
        new_state.snake.alive     = self.snake.alive
        new_state.snake.growing   = self.snake.growing

        if self.food.position:
            new_state.food.position = self.food.position.copy()

        new_state.score     = self.score
        new_state.moves     = self.moves
        new_state.game_over = self.game_over
        new_state.won       = self.won

        return new_state
