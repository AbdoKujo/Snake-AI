"""Renderer Pygame — Neural Grid cyberpunk theme."""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
import math
import pygame

if TYPE_CHECKING:
    from game.game_state import GameState
    from game.snake import Snake
    from game.food import Food
    from game.point import Point
    from ai.base_agent import BaseAgent
    from ai.astar_agent import AStarAgent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ---------------------------------------------------------------------------
# Default Neural Grid palette
# ---------------------------------------------------------------------------

_DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "background":     (6, 6, 18),
    "bg_panel":       (12, 14, 32),
    "bg_card":        (16, 18, 44),
    "grid_line":      (18, 20, 48),
    "snake_head":     (0, 255, 160),
    "snake_body":     (0, 200, 120),
    "snake_tail":     (0, 130, 80),
    "snake_outline":  (0, 75, 48),
    "food":           (255, 60, 90),
    "food_glow":      (255, 110, 135),
    "accent":         (0, 255, 160),
    "accent_dim":     (0, 100, 65),
    "astar_color":    (60, 160, 255),
    "dqn_color":      (210, 80, 255),
    "human_color":    (255, 210, 50),
    "text":           (200, 210, 235),
    "text_dim":       (75, 90, 125),
    "text_bright":    (255, 255, 255),
    "hud_bg":         (9, 9, 25),
    "button_bg":      (16, 18, 44),
    "button_hover":   (24, 28, 64),
    "button_border":  (38, 44, 100),
    "score_bg":       (20, 22, 55),
    "win":            (0, 255, 130),
    "lose":           (255, 55, 85),
}


class Renderer:
    """
    Handles all Pygame rendering for Snake Game AI — Neural Grid theme.
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        cell_size: int = 20,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> None:
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.cell_size   = cell_size

        self.game_width   = grid_width  * cell_size
        self.game_height  = grid_height * cell_size
        self.hud_height   = 70
        self.window_width  = self.game_width
        self.window_height = self.game_height + self.hud_height

        self.colors = dict(_DEFAULT_COLORS)
        if colors:
            self.colors.update(colors)

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Snake Game AI")

        # All fonts use Consolas
        self.font_score   = pygame.font.SysFont("consolas", 26, bold=True)
        self.font_label   = pygame.font.SysFont("consolas", 11, bold=False)
        self.font_badge   = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_time    = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_over_h  = pygame.font.SysFont("consolas", 38, bold=True)
        self.font_over_s  = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_over_m  = pygame.font.SysFont("consolas", 14, bold=False)
        self.font_key     = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_eps     = pygame.font.SysFont("consolas", 11, bold=False)

        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, game_state, agent=None) -> None:
        """Main render call — draws the full frame."""
        self.screen.fill(self.colors["background"])

        self.draw_grid()

        if agent is not None and hasattr(agent, "get_current_path"):
            self.draw_path(agent.get_current_path())

        self.draw_food(game_state.food)
        self.draw_snake(game_state.snake)
        self.draw_hud(game_state, agent)

        if game_state.game_over:
            self.draw_game_over(game_state)

        pygame.display.flip()

    def quit(self) -> None:
        """Shut down Pygame cleanly."""
        pygame.quit()

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def draw_grid(self) -> None:
        """Draw subtle grid lines."""
        col = self.colors["grid_line"]
        for x in range(0, self.game_width + 1, self.cell_size):
            pygame.draw.line(self.screen, col, (x, 0), (x, self.game_height))
        for y in range(0, self.game_height + 1, self.cell_size):
            pygame.draw.line(self.screen, col, (0, y), (self.game_width, y))

    # ------------------------------------------------------------------
    # Snake
    # ------------------------------------------------------------------

    def draw_snake(self, snake) -> None:
        """Draw the snake with a gradient body and eyes on the head."""
        body = snake.body
        n = len(body)

        for i, segment in enumerate(body):
            px = segment.x * self.cell_size
            py = segment.y * self.cell_size

            if i == 0:
                color = self.colors["snake_head"]
            else:
                t = i / max(n - 1, 1)
                color = _lerp_color((0, 200, 120), (0, 130, 80), t)

            rect = pygame.Rect(px + 1, py + 1, self.cell_size - 2, self.cell_size - 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.colors["snake_outline"], rect, 1, border_radius=4)

            if i == 0:
                self._draw_snake_eyes(px, py, snake.direction)

    def _draw_snake_eyes(self, x: int, y: int, direction) -> None:
        from game.direction import Direction

        eye_radius = 3
        cx = x + self.cell_size // 2
        cy = y + self.cell_size // 2
        offset = 4

        if direction == Direction.UP:
            eye1 = (cx - offset, cy - offset)
            eye2 = (cx + offset, cy - offset)
        elif direction == Direction.DOWN:
            eye1 = (cx - offset, cy + offset)
            eye2 = (cx + offset, cy + offset)
        elif direction == Direction.LEFT:
            eye1 = (cx - offset, cy - offset)
            eye2 = (cx - offset, cy + offset)
        else:  # RIGHT
            eye1 = (cx + offset, cy - offset)
            eye2 = (cx + offset, cy + offset)

        pygame.draw.circle(self.screen, (255, 255, 255), eye1, eye_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), eye2, eye_radius)
        pygame.draw.circle(self.screen, (0, 0, 0), eye1, eye_radius - 1)
        pygame.draw.circle(self.screen, (0, 0, 0), eye2, eye_radius - 1)

    # ------------------------------------------------------------------
    # Food
    # ------------------------------------------------------------------

    def draw_food(self, food) -> None:
        """Draw animated pulsing food with glow."""
        if food.position is None:
            return

        t_sec = pygame.time.get_ticks() / 1000.0
        pulse = math.sin(t_sec * 4.0)

        px = food.position.x * self.cell_size
        py = food.position.y * self.cell_size
        center = (px + self.cell_size // 2, py + self.cell_size // 2)

        # Glow on SRCALPHA surface
        glow_radius = self.cell_size // 2 + 3 + int(2 * pulse)
        glow_surf = pygame.Surface((self.cell_size * 3, self.cell_size * 3), pygame.SRCALPHA)
        glow_center = (self.cell_size * 3 // 2, self.cell_size * 3 // 2)
        glow_col = (*self.colors["food_glow"], 80)
        pygame.draw.circle(glow_surf, glow_col, glow_center, glow_radius)
        self.screen.blit(glow_surf, (center[0] - self.cell_size * 3 // 2,
                                      center[1] - self.cell_size * 3 // 2))

        # Inner food circle
        inner_radius = self.cell_size // 2 - 3
        pygame.draw.circle(self.screen, self.colors["food"], center, inner_radius)

    # ------------------------------------------------------------------
    # A* Path
    # ------------------------------------------------------------------

    def draw_path(self, path: List) -> None:
        """Draw A* path as fading blue dots."""
        if not path or len(path) < 2:
            return

        path_surf = pygame.Surface((self.game_width, self.game_height), pygame.SRCALPHA)
        path_col_base = self.colors["astar_color"]
        n = len(path) - 1  # skip head

        for i, point in enumerate(path[1:], 1):
            fade = max(0.15, 1.0 - i / max(n, 1))
            alpha = int(120 * fade)
            col = (*path_col_base, alpha)
            px = point.x * self.cell_size + self.cell_size // 2 - 2
            py = point.y * self.cell_size + self.cell_size // 2 - 2
            pygame.draw.rect(path_surf, col, (px, py, 4, 4))

        self.screen.blit(path_surf, (0, 0))

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def draw_hud(self, game_state, agent=None) -> None:
        """Draw the 60px HUD bar with three sections."""
        hud_y = self.game_height
        C = self.colors

        # HUD background
        hud_rect = pygame.Rect(0, hud_y, self.window_width, self.hud_height)
        pygame.draw.rect(self.screen, C["hud_bg"], hud_rect)

        # Thin accent line at top of HUD
        accent_line = pygame.Surface((self.window_width, 1), pygame.SRCALPHA)
        accent_line.fill((*C["accent"], 60))
        self.screen.blit(accent_line, (0, hud_y))

        # Vertical dividers
        for div_x in (210, 430):
            div_surf = pygame.Surface((1, self.hud_height), pygame.SRCALPHA)
            div_surf.fill((*C["button_border"], 200))
            self.screen.blit(div_surf, (div_x, hud_y))

        # ---- Section 1: Score (x 0..210) ----
        score = getattr(game_state, "score", 0)
        length = getattr(game_state.snake, "length", len(game_state.snake.body))

        score_surf = self.font_score.render(str(score), True, C["accent"])
        self.screen.blit(score_surf, (14, hud_y + 10))

        sub_text = f"SCORE  \u00b7  LEN {length}"
        sub_surf = self.font_label.render(sub_text, True, C["text_dim"])
        self.screen.blit(sub_surf, (14, hud_y + 48))

        # ---- Section 2: Agent info (x 210..430, centred at 320) ----
        cx2 = 320

        # Determine agent type and paused state
        paused = getattr(game_state, "paused", False)
        agent_name = ""
        agent_col = C["accent"]
        epsilon = None

        if paused:
            paused_surf = self.font_badge.render("[ PAUSED ]", True, C["human_color"])
            self.screen.blit(paused_surf, paused_surf.get_rect(center=(cx2, hud_y + 16)))
        else:
            if agent is not None:
                aname = getattr(agent, "name", "").lower()
                if "astar" in aname or "a*" in aname or "a_star" in aname:
                    agent_name = "A* AGENT"
                    agent_col  = C["astar_color"]
                elif "dqn" in aname or "deep" in aname:
                    agent_name = "DQN AGENT"
                    agent_col  = C["dqn_color"]
                    if hasattr(agent, "get_stats"):
                        stats = agent.get_stats()
                        epsilon = stats.get("epsilon", None)
                else:
                    agent_name = getattr(agent, "name", "AGENT").upper()
            else:
                agent_name = "HUMAN"
                agent_col  = C["human_color"]

            # Badge
            badge_text_surf = self.font_badge.render(agent_name, True, C["text_bright"])
            badge_w = badge_text_surf.get_width() + 24
            badge_h = 24
            badge_rect = pygame.Rect(cx2 - badge_w // 2, hud_y + 4, badge_w, badge_h)
            pygame.draw.rect(self.screen, agent_col, badge_rect, border_radius=4)
            self.screen.blit(badge_text_surf, badge_text_surf.get_rect(center=badge_rect.center))

            # DQN epsilon bar
            if epsilon is not None:
                eps_bar_w = 80
                eps_bar_h = 5
                eps_bar_x = cx2 - eps_bar_w // 2
                eps_bar_y = hud_y + 36
                # Background
                pygame.draw.rect(self.screen, C["button_border"],
                                  (eps_bar_x, eps_bar_y, eps_bar_w, eps_bar_h), border_radius=2)
                # Fill
                fill_w = max(1, int(eps_bar_w * epsilon))
                pygame.draw.rect(self.screen, C["dqn_color"],
                                  (eps_bar_x, eps_bar_y, fill_w, eps_bar_h), border_radius=2)
                # Epsilon label
                eps_label = self.font_eps.render(f"\u03b5 {epsilon:.2f}", True, C["text_dim"])
                self.screen.blit(eps_label, (eps_bar_x + eps_bar_w + 4, eps_bar_y - 1))

        # ---- Section 3: Time + Moves (x 430..600, right-aligned) ----
        elapsed = 0
        if hasattr(game_state, "get_elapsed_time"):
            elapsed = game_state.get_elapsed_time()
        moves = getattr(game_state, "moves", 0)

        total_secs = int(elapsed)
        mm = total_secs // 60
        ss = total_secs % 60
        time_str = f"{mm:02d}:{ss:02d}"

        time_surf = self.font_time.render(time_str, True, C["text_bright"])
        self.screen.blit(time_surf, (self.window_width - time_surf.get_width() - 12, hud_y + 8))

        moves_str = f"MOVES: {moves}"
        moves_surf = self.font_label.render(moves_str, True, C["text_dim"])
        self.screen.blit(moves_surf, (self.window_width - moves_surf.get_width() - 12, hud_y + 38))

    # ------------------------------------------------------------------
    # Game Over overlay
    # ------------------------------------------------------------------

    def draw_game_over(self, game_state) -> None:
        """Draw a styled game-over / win overlay panel."""
        C = self.colors

        # Semi-transparent full overlay
        overlay = pygame.Surface((self.game_width, self.game_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        won = getattr(game_state, "won", False)
        if won:
            headline  = "YOU WIN!"
            hl_color  = C["win"]
            brd_color = C["win"]
        else:
            headline  = "GAME OVER"
            hl_color  = C["lose"]
            brd_color = C["lose"]

        panel_w = 380
        panel_h = 220
        panel_x = (self.game_width  - panel_w) // 2
        panel_y = (self.game_height - panel_h) // 2

        # Panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, C["bg_card"], panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, brd_color,   panel_rect, 2, border_radius=10)

        cx = panel_x + panel_w // 2

        # Headline
        hl_surf = self.font_over_h.render(headline, True, hl_color)
        self.screen.blit(hl_surf, hl_surf.get_rect(center=(cx, panel_y + 42)))

        # Final score
        score = getattr(game_state, "score", 0)
        sc_surf = self.font_over_s.render(f"FINAL SCORE: {score}", True, C["text_bright"])
        self.screen.blit(sc_surf, sc_surf.get_rect(center=(cx, panel_y + 92)))

        # Stats row
        length = len(game_state.snake.body)
        moves  = getattr(game_state, "moves", 0)
        st_surf = self.font_over_m.render(f"LENGTH: {length}   \u00b7   MOVES: {moves}", True, C["text_dim"])
        self.screen.blit(st_surf, st_surf.get_rect(center=(cx, panel_y + 128)))

        # Key hint badges
        self._draw_key_badge(self.screen, "R", "RESTART", cx - 90, panel_y + 166, C)
        self._draw_key_badge(self.screen, "ESC", "MENU",   cx + 80, panel_y + 166, C)

    def _draw_key_badge(
        self,
        surf: pygame.Surface,
        key: str,
        label: str,
        cx: int,
        cy: int,
        C: dict,
    ) -> None:
        """Draw a small key-hint badge like [ R ] RESTART."""
        key_surf   = self.font_key.render(f"[ {key} ]", True, C["accent"])
        label_surf = self.font_key.render(f" {label}", True, C["text_dim"])

        total_w = key_surf.get_width() + label_surf.get_width() + 16
        badge_h = key_surf.get_height() + 8
        badge_x = cx - total_w // 2
        badge_y = cy - badge_h // 2

        badge_rect = pygame.Rect(badge_x, badge_y, total_w, badge_h)
        pygame.draw.rect(surf, C["bg_card"],        badge_rect, border_radius=4)
        pygame.draw.rect(surf, C["button_border"],  badge_rect, 1, border_radius=4)

        surf.blit(key_surf,   (badge_x + 8, badge_y + 4))
        surf.blit(label_surf, (badge_x + 8 + key_surf.get_width(), badge_y + 4))

    # ------------------------------------------------------------------
    # Legacy compat
    # ------------------------------------------------------------------

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not getattr(self, '_fullscreen', False)
        if self._fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))

    def update(self) -> None:
        pygame.display.flip()
