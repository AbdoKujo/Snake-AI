"""Vue écran partagé — A* vs DQN en temps réel."""

from __future__ import annotations
import math
from typing import Optional
import pygame

from config import GRID_WIDTH, GRID_HEIGHT, DQN_CONFIG, GAME_SPEED_VERSUS, SPLIT_CELL_SIZE
from game.game_state import GameState
from game.direction import Direction
from ai.astar_agent import AStarAgent
from ai.dqn_agent import DQNAgent

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_BG      = (6,   6,  18)
_PANEL   = (8,   8,  22)
_DIVIDER = (16,  18,  50)
_HEADER  = (10,  10,  28)
_STATS   = (9,   9,  25)
_ASTAR   = (60,  160, 255)
_DQN     = (210,  80, 255)
_ACCENT  = (0,  255, 160)
_TEXT    = (200, 210, 235)
_DIM     = (75,  90, 125)
_GRID    = (18,  20,  48)
_FOOD    = (255,  60,  90)
_FOOD_G  = (255, 110, 135)
_S_HEAD  = (0,  255, 160)
_S_BODY  = (0,  200, 120)
_S_TAIL  = (0,  130,  80)
_S_OUT   = (0,   75,  48)
_WIN     = (0,  255, 130)
_LOSE    = (255,  55,  85)
_BORDER  = (38,  44, 100)


def _lerp(c1: tuple, c2: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class SplitScreenView:
    """
    Displays A* and DQN agents playing simultaneously in a split-screen window.
    Left panel = A* (with path visualisation), Right panel = DQN.
    Each game auto-restarts after a short delay on game-over so the viewer
    can compare the two agents continuously.
    """

    HEADER_H    = 55
    STATS_H     = 65
    GAP_W       = 40     # Width of centre divider
    DEATH_DELAY = 1.8    # Seconds to display score before auto-reset

    def __init__(self, db_manager=None) -> None:
        self.db_manager = db_manager
        self.cs = SPLIT_CELL_SIZE     # cell size (15)
        self.gw = GRID_WIDTH          # 30
        self.gh = GRID_HEIGHT         # 25

        self.panel_w = self.gw * self.cs                         # 450
        self.panel_h = self.gh * self.cs                         # 375
        self.total_w = self.panel_w * 2 + self.GAP_W             # 940
        self.total_h = self.HEADER_H + self.panel_h + self.STATS_H  # 495
        self._fullscreen = False

        pygame.display.set_caption("SNAKE GAME AI  —  A* vs DQN")
        self.screen = pygame.display.set_mode((self.total_w, self.total_h))

        # Fonts
        self.f_title = pygame.font.SysFont("consolas", 18, bold=True)
        self.f_score = pygame.font.SysFont("consolas", 26, bold=True)
        self.f_label = pygame.font.SysFont("consolas", 13, bold=False)
        self.f_vs    = pygame.font.SysFont("consolas", 16, bold=True)
        self.f_stat  = pygame.font.SysFont("consolas", 14, bold=True)
        self.f_hint  = pygame.font.SysFont("consolas", 12, bold=False)
        self.f_go    = pygame.font.SysFont("consolas", 22, bold=True)
        self.f_badge = pygame.font.SysFont("consolas", 11, bold=True)

        # Game states
        self.state_astar = GameState(self.gw, self.gh)
        self.state_dqn   = GameState(self.gw, self.gh)

        # Agents
        self.astar = AStarAgent()
        self.dqn   = DQNAgent(**DQN_CONFIG)
        try:
            self.dqn.load_model("saved_models/dqn_model.pt")
            self.dqn.set_eval_mode()
            print("[SplitScreen] Loaded pre-trained DQN model.")
        except FileNotFoundError:
            self.dqn.epsilon = 0.15
            print("[SplitScreen] No DQN model found — using random policy.")

        # Cumulative session stats
        self.astar_stats = {"rounds": 0, "total": 0, "best": 0, "last": 0}
        self.dqn_stats   = {"rounds": 0, "total": 0, "best": 0, "last": 0}

        # Death timers (float seconds since game-over)
        self._astar_dt = 0.0
        self._dqn_dt   = 0.0

        # Panel surfaces (drawn into then blitted to screen)
        self._lsurf = pygame.Surface((self.panel_w, self.panel_h))
        self._rsurf = pygame.Surface((self.panel_w, self.panel_h))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop. Returns when the user presses ESC or closes the window."""
        clock = pygame.time.Clock()
        prev  = pygame.time.get_ticks()

        while True:
            now = pygame.time.get_ticks()
            dt  = min((now - prev) / 1000.0, 0.1)
            prev = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    elif event.key == pygame.K_r:
                        self._reset_both()

            self._step_agent(self.state_astar, self.astar,
                             self.astar_stats, "_astar_dt", dt)
            self._step_agent(self.state_dqn,   self.dqn,
                             self.dqn_stats,   "_dqn_dt",   dt)

            self._render()
            clock.tick(GAME_SPEED_VERSUS)

    # ------------------------------------------------------------------
    # Game-logic helpers
    # ------------------------------------------------------------------

    def _step_agent(
        self,
        state: GameState,
        agent,
        stats: dict,
        timer_attr: str,
        dt: float,
    ) -> None:
        if not state.game_over:
            action = agent.get_action(state)
            state.update(action)
        else:
            elapsed = getattr(self, timer_attr) + dt
            setattr(self, timer_attr, elapsed)
            if elapsed >= self.DEATH_DELAY:
                score = state.score
                stats["rounds"] += 1
                stats["total"]  += score
                stats["best"]    = max(stats["best"], score)
                stats["last"]    = score
                state.reset()
                setattr(self, timer_attr, 0.0)

    def _reset_both(self) -> None:
        self.state_astar.reset()
        self.state_dqn.reset()
        self._astar_dt = self._dqn_dt = 0.0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        t = pygame.time.get_ticks() / 1000.0
        self.screen.fill(_BG)

        self._draw_header(t)

        panel_top = self.HEADER_H

        # Left panel (A*)
        self._draw_panel(self._lsurf, self.state_astar, self.astar,
                         self._astar_dt, show_path=True, t=t)
        self.screen.blit(self._lsurf, (0, panel_top))

        # Central divider
        self._draw_divider(self.panel_w, panel_top, t)

        # Right panel (DQN)
        self._draw_panel(self._rsurf, self.state_dqn, self.dqn,
                         self._dqn_dt, show_path=False, t=t)
        self.screen.blit(self._rsurf, (self.panel_w + self.GAP_W, panel_top))

        self._draw_stats_bar(t)
        pygame.display.flip()

    # -- Header -----------------------------------------------------------

    def _draw_header(self, t: float) -> None:
        hdr_rect = pygame.Rect(0, 0, self.total_w, self.HEADER_H)
        pygame.draw.rect(self.screen, _HEADER, hdr_rect)
        # Bottom accent line
        pygame.draw.line(self.screen, _ACCENT,
                         (0, self.HEADER_H - 1), (self.total_w, self.HEADER_H - 1), 1)

        pulse = 0.7 + 0.3 * math.sin(t * 2.0)
        mid_y = self.HEADER_H // 2

        # A* label — left
        a_col  = tuple(int(c * pulse) for c in _ASTAR)
        a_surf = self.f_title.render("◆  A* AGENT", True, a_col)
        self.screen.blit(a_surf, (18, mid_y - a_surf.get_height() // 2))

        # DQN label — right
        d_col  = tuple(int(c * pulse) for c in _DQN)
        d_surf = self.f_title.render("DQN AGENT  ●", True, d_col)
        self.screen.blit(d_surf,
                         (self.total_w - d_surf.get_width() - 18,
                          mid_y - d_surf.get_height() // 2))

        # VS — centre
        vs_col  = _lerp(_ASTAR, _DQN, 0.5 + 0.5 * math.sin(t * 1.5))
        vs_surf = self.f_vs.render("VS", True, vs_col)
        self.screen.blit(vs_surf,
                         vs_surf.get_rect(center=(self.total_w // 2, mid_y)))

        # Controls hint — top-right corner
        hint = self.f_hint.render(
            "F11 fullscreen  ·  R restart  ·  ESC menu", True, _DIM)
        self.screen.blit(hint, (self.total_w - hint.get_width() - 10, 4))

    # -- Central divider --------------------------------------------------

    def _draw_divider(self, gap_x: int, oy: int, t: float) -> None:
        div_rect = pygame.Rect(gap_x, oy, self.GAP_W, self.panel_h)
        pygame.draw.rect(self.screen, _DIVIDER, div_rect)

        cx = gap_x + self.GAP_W // 2

        # Animated centre line
        pulse = 0.4 + 0.3 * math.sin(t * 3.0)
        line_col = tuple(int(c * pulse) for c in _ACCENT)
        pygame.draw.line(self.screen, line_col,
                         (cx, oy + 8), (cx, oy + self.panel_h - 8), 1)

        # Live scores centred in the gap
        half_y = oy + self.panel_h // 2
        a_sc = self.f_stat.render(str(self.state_astar.score), True, _ASTAR)
        d_sc = self.f_stat.render(str(self.state_dqn.score),   True, _DQN)
        lbl  = self.f_hint.render("score", True, _DIM)

        self.screen.blit(a_sc,  a_sc.get_rect(center=(cx, half_y - 22)))
        self.screen.blit(lbl,   lbl.get_rect(center=(cx, half_y)))
        self.screen.blit(d_sc,  d_sc.get_rect(center=(cx, half_y + 22)))

    # -- Stats bar --------------------------------------------------------

    def _draw_stats_bar(self, t: float) -> None:
        sy = self.HEADER_H + self.panel_h
        pygame.draw.rect(self.screen, _STATS,
                         pygame.Rect(0, sy, self.total_w, self.STATS_H))
        pygame.draw.line(self.screen, _ACCENT, (0, sy), (self.total_w, sy), 1)

        def _side(stats: dict, x: int, color: tuple, label: str) -> None:
            # Colored agent badge
            badge_surf = self.f_badge.render(label, True, (0, 0, 0))
            bw = badge_surf.get_width() + 14
            bh = 20
            badge_rect = pygame.Rect(x, sy + 8, bw, bh)
            pygame.draw.rect(self.screen, color, badge_rect, border_radius=3)
            self.screen.blit(badge_surf, badge_surf.get_rect(center=badge_rect.center))

            # Row of stats
            avg = stats["total"] / max(stats["rounds"], 1)
            row = (f"  RND {stats['rounds']}"
                   f"   AVG {avg:.1f}"
                   f"   BEST {stats['best']}"
                   f"   LAST {stats['last']}")
            row_surf = self.f_label.render(row, True, _DIM)
            self.screen.blit(row_surf, (x, sy + 36))

        _side(self.astar_stats, 10, _ASTAR, "A* AGENT")

        # Centre hint
        hint = self.f_hint.render("R — restart both  ·  ESC — menu", True, _DIM)
        self.screen.blit(hint,
                         hint.get_rect(center=(self.total_w // 2,
                                               sy + self.STATS_H // 2)))

        _side(self.dqn_stats, self.panel_w + self.GAP_W + 10, _DQN, "DQN")

    # -- Game panel -------------------------------------------------------

    def _draw_panel(
        self,
        surf: pygame.Surface,
        state: GameState,
        agent,
        death_timer: float,
        show_path: bool,
        t: float,
    ) -> None:
        cs = self.cs
        surf.fill(_PANEL)

        # Grid
        for gx in range(0, self.panel_w, cs):
            pygame.draw.line(surf, _GRID, (gx, 0), (gx, self.panel_h))
        for gy in range(0, self.panel_h, cs):
            pygame.draw.line(surf, _GRID, (0, gy), (self.panel_w, gy))

        # A* path
        if show_path and hasattr(agent, "get_current_path"):
            path = agent.get_current_path()
            if path and len(path) > 1:
                path_surf = pygame.Surface((self.panel_w, self.panel_h),
                                           pygame.SRCALPHA)
                n_steps = len(path) - 1
                for i, p in enumerate(path[1:], 1):
                    fade  = max(0.1, 1.0 - i / max(n_steps, 1))
                    alpha = int(110 * fade)
                    rx = p.x * cs + 3
                    ry = p.y * cs + 3
                    pygame.draw.rect(path_surf, (*_ASTAR, alpha),
                                     (rx, ry, cs - 6, cs - 6), border_radius=2)
                surf.blit(path_surf, (0, 0))

        # Food (animated pulse)
        if state.food.position:
            fx = state.food.position.x * cs + cs // 2
            fy = state.food.position.y * cs + cs // 2
            pulse_r = cs // 2 + 1 + int(1.5 * math.sin(t * 5.0))
            glow_s  = pygame.Surface((cs * 2, cs * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (*_FOOD_G, 70), (cs, cs), pulse_r + 2)
            surf.blit(glow_s, (fx - cs, fy - cs))
            pygame.draw.circle(surf, _FOOD, (fx, fy), max(2, cs // 2 - 2))

        # Snake
        body = state.snake.body
        n    = len(body)
        for i, seg in enumerate(body):
            sx = seg.x * cs + 1
            sy = seg.y * cs + 1
            col = _S_HEAD if i == 0 else _lerp(_S_BODY, _S_TAIL, i / max(n - 1, 1))
            rect = pygame.Rect(sx, sy, cs - 2, cs - 2)
            pygame.draw.rect(surf, col,   rect, border_radius=3)
            pygame.draw.rect(surf, _S_OUT, rect, 1, border_radius=3)
            if i == 0:
                self._draw_eyes(surf, sx, sy, cs, state.snake.direction)

        # Game-over overlay
        if state.game_over:
            ov = pygame.Surface((self.panel_w, self.panel_h), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 145))
            surf.blit(ov, (0, 0))

            col = _WIN if state.won else _LOSE
            txt = "WIN!" if state.won else "GAME OVER"
            go  = self.f_go.render(txt, True, col)
            surf.blit(go, go.get_rect(center=(self.panel_w // 2,
                                               self.panel_h // 2 - 16)))

            sc = self.f_stat.render(f"SCORE: {state.score}", True, _TEXT)
            surf.blit(sc, sc.get_rect(center=(self.panel_w // 2,
                                               self.panel_h // 2 + 12)))

            remaining = max(0.0, self.DEATH_DELAY - death_timer)
            cnt = self.f_hint.render(f"restart in {remaining:.1f}s ...", True, _DIM)
            surf.blit(cnt, cnt.get_rect(center=(self.panel_w // 2,
                                                 self.panel_h // 2 + 34)))

    # -- Eyes -------------------------------------------------------------

    def _draw_eyes(
        self,
        surf: pygame.Surface,
        sx: int, sy: int,
        cs: int,
        direction,
    ) -> None:
        cx, cy = sx + cs // 2, sy + cs // 2
        off = max(2, cs // 5)
        er  = max(1, cs // 8)
        if direction == Direction.UP:
            e1, e2 = (cx - off, cy - off), (cx + off, cy - off)
        elif direction == Direction.DOWN:
            e1, e2 = (cx - off, cy + off), (cx + off, cy + off)
        elif direction == Direction.LEFT:
            e1, e2 = (cx - off, cy - off), (cx - off, cy + off)
        else:
            e1, e2 = (cx + off, cy - off), (cx + off, cy + off)
        for e in (e1, e2):
            pygame.draw.circle(surf, (255, 255, 255), e, er)
            pygame.draw.circle(surf, (0, 0, 0),       e, max(1, er - 1))

    # -- Fullscreen -------------------------------------------------------

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.total_w, self.total_h))
        # Panel surfaces are fixed size — re-allocate cleanly
        self._lsurf = pygame.Surface((self.panel_w, self.panel_h))
        self._rsurf = pygame.Surface((self.panel_w, self.panel_h))
