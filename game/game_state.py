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

        self.score     = 0
        self.moves     = 0
        self.game_over = False
        self.won       = False
        self.start_time = time.time()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, action: Optional[Direction] = None) -> Tuple[bool, float]:
        """
        Met à jour l'état du jeu après un mouvement.

        Returns:
            Tuple (done, reward).
        """
        if self.game_over:
            return True, 0.0

        if action is not None:
            self.snake.change_direction(action)

        self.snake.move()
        self.moves += 1

        # Collision → death.
        # Penalty scales with length: longer snake = bigger loss (more progress gone).
        if self.snake.check_collision(self.grid_width, self.grid_height):
            self.snake.alive = False
            self.game_over   = True
            death_penalty    = -10.0 - 0.1 * len(self.snake.body)
            return True, death_penalty

        # Dynamic step penalty: inversely proportional to snake length.
        # Short snake (len 3)  → -0.010  (same pressure as before)
        # Medium snake (len 10) → -0.003  (less panic, more room to plan)
        # Long snake (len 30)  → -0.001  (almost no pressure — path is complex)
        snake_len = len(self.snake.body)
        reward    = -0.03 / max(snake_len, 3)

        if self.food.is_at(self.snake.head):
            old_len = snake_len
            self.snake.grow()
            self.score += self.food.value
            reward = 10.0

            # Milestone bonus: +5 every time the snake crosses a multiple of 10.
            # Provides gradient signal throughout training (not just at unreachable win).
            new_len = len(self.snake.body)
            if (new_len // 10) > (old_len // 10):
                reward += 5.0

            # Victory: grid is full
            if not self.food.spawn(self.snake.get_body_set()):
                self.won       = True
                self.game_over = True
                reward         = 100.0

        return False, reward

    # ------------------------------------------------------------------
    # State representation — 14 features
    # ------------------------------------------------------------------

    def get_state_representation(self) -> np.ndarray:
        """
        Build a 14-dimensional state vector for the DQN.

        Features:
          [0-2]  danger flags    (ahead, left, right) — binary
          [3-5]  open space      (ahead, left, right) — BFS flood-fill, normalised
          [6-9]  direction       one-hot (UP, DOWN, LEFT, RIGHT)
          [10-11] food offset    (dx, dy) normalised to [-1, 1]
          [12]   snake length    normalised to [0, 1]
          [13]   can_reach_tail  BFS boolean — prevents self-trapping

        Returns:
            np.ndarray of shape (14,), dtype float32.
        """
        head      = self.snake.head
        direction = self.snake.direction
        ddx, ddy  = direction.to_vector()
        gw, gh    = self.grid_width, self.grid_height
        hx, hy    = head.x, head.y

        # Build obstacle set ONCE as (x,y) tuples — shared by all 4 BFS calls.
        # Avoids creating 4 separate Python sets from get_body_set().
        body_tuples: Set[tuple] = {(p.x, p.y) for p in self.snake.body}

        # Adjacent cell coordinates
        ax, ay = hx + ddx, hy + ddy          # ahead
        if direction == Direction.UP:
            lx, ly, rx, ry = hx - 1, hy,     hx + 1, hy
        elif direction == Direction.DOWN:
            lx, ly, rx, ry = hx + 1, hy,     hx - 1, hy
        elif direction == Direction.LEFT:
            lx, ly, rx, ry = hx,     hy + 1, hx,     hy - 1
        else:  # RIGHT
            lx, ly, rx, ry = hx,     hy - 1, hx,     hy + 1

        # [0-2] Danger: out-of-bounds OR body
        def _danger(x: int, y: int) -> float:
            return 1.0 if (x < 0 or x >= gw or y < 0 or y >= gh
                           or (x, y) in body_tuples) else 0.0

        danger_ahead = _danger(ax, ay)
        danger_left  = _danger(lx, ly)
        danger_right = _danger(rx, ry)

        # [3-5] Flood-fill open space — all three share the same obstacle set
        total_cells = gw * gh
        space_ahead = self._bfs_count(ax, ay, body_tuples, gw, gh) / total_cells
        space_left  = self._bfs_count(lx, ly, body_tuples, gw, gh) / total_cells
        space_right = self._bfs_count(rx, ry, body_tuples, gw, gh) / total_cells

        # [6-9] Direction one-hot
        dir_up    = 1.0 if direction == Direction.UP    else 0.0
        dir_down  = 1.0 if direction == Direction.DOWN  else 0.0
        dir_left  = 1.0 if direction == Direction.LEFT  else 0.0
        dir_right = 1.0 if direction == Direction.RIGHT else 0.0

        # [10-11] Food offset normalised to [-1, 1]
        food_pos = self.food.position
        food_dx  = (food_pos.x - hx) / gw
        food_dy  = (food_pos.y - hy) / gh

        # [12] Snake length normalised
        snake_length_norm = len(self.snake.body) / total_cells

        # [13] Can head reach tail? Exclude tail from obstacles (it vacates next step)
        tail = self.snake.body[-1]
        tail_t = (tail.x, tail.y)
        tail_obstacles = body_tuples - {tail_t}
        can_reach_tail = 1.0 if self._bfs_reaches(
            hx, hy, tail.x, tail.y, tail_obstacles, gw, gh) else 0.0

        return np.array([
            danger_ahead, danger_left, danger_right,
            space_ahead,  space_left,  space_right,
            dir_up, dir_down, dir_left, dir_right,
            food_dx, food_dy,
            snake_length_norm,
            can_reach_tail,
        ], dtype=np.float32)

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
