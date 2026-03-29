"""Vue de comparaison des performances — Neural Grid cyberpunk theme."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import math
import pygame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_BG         = (6,  6,  18)
_CARD_BG    = (16, 18, 44)
_BORDER     = (38, 44, 100)
_ACCENT     = (0,  255, 160)
_TEXT       = (200, 210, 235)
_TEXT_DIM   = (75,  90,  125)
_TEXT_WHITE = (255, 255, 255)
_ASTAR_COL  = (60,  160, 255)
_DQN_COL    = (210,  80, 255)
_HUMAN_COL  = (255, 210,  50)
_HUD_BG     = (9,   9,  25)

_AGENT_META = {
    "human": {
        "label":  "HUMAN",
        "color":  _HUMAN_COL,
        "bar":    "#ffd232",
    },
    "astar": {
        "label":  "A* AGENT",
        "color":  _ASTAR_COL,
        "bar":    "#3ca0ff",
    },
    "dqn": {
        "label":  "DQN",
        "color":  _DQN_COL,
        "bar":    "#d250ff",
    },
}

_AGENT_ORDER = ["human", "astar", "dqn"]

# Card layout
_CARD_W   = 178
_CARD_H   = 200
_CARD_Y   = 70
_CARD_GAP = 8
# x positions: 14, 200, 386
_CARD_XS  = [14, 14 + _CARD_W + _CARD_GAP, 14 + 2 * (_CARD_W + _CARD_GAP)]


def _lerp_color(c1: Tuple, c2: Tuple, t: float) -> Tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class ComparisonView:
    """
    Displays comparative performance statistics between agents.
    Neural Grid cyberpunk theme, 600x560 window.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        self.screen_width  = screen_width
        self.screen_height = screen_height

        if not pygame.get_init():
            pygame.init()

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Snake Game AI - Statistics")

        # Fonts — all Consolas
        self.font_title      = pygame.font.SysFont("consolas", 32, bold=True)
        self.font_card_label = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_stat_val   = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_stat_lbl   = pygame.font.SysFont("consolas", 11, bold=False)
        self.font_nodata     = pygame.font.SysFont("consolas", 13, bold=False)
        self.font_back       = pygame.font.SysFont("consolas", 14, bold=False)
        self.font_footer     = pygame.font.SysFont("consolas", 12, bold=False)

        self._back_hover: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, statistics: Dict[str, Optional[Dict]]) -> None:
        """
        Display the statistics screen until the user presses ESC or clicks Back.

        Args:
            statistics: dict with keys 'human', 'astar', 'dqn'; each value is
                        a dict with total_games, avg_score, max_score, win_rate,
                        or None if no data.
        """
        clock = pygame.time.Clock()
        chart_surf = self._create_charts(statistics)

        while True:
            dt = clock.tick(60) / 1000.0
            mouse_pos   = pygame.mouse.get_pos()
            mouse_click = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_click = True

            # Back button hover + click
            back_rect = self._back_button_rect()
            if back_rect.collidepoint(mouse_pos):
                self._back_hover = min(1.0, self._back_hover + dt * 6.0)
                if mouse_click:
                    return
            else:
                self._back_hover = max(0.0, self._back_hover - dt * 6.0)

            self._draw(statistics, chart_surf)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _back_button_rect(self) -> pygame.Rect:
        btn_w = 160
        btn_h = 34
        btn_x = (self.screen_width - btn_w) // 2
        btn_y = 510
        return pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, statistics: Dict[str, Optional[Dict]], chart_surf: Optional[pygame.Surface]) -> None:
        self.screen.fill(_BG)

        # Title with glow
        self._draw_glow_title("PERFORMANCE STATS", (self.screen_width // 2, 36))

        # Divider at y=60
        self._draw_divider(60)

        # Agent cards
        for col_idx, agent_key in enumerate(_AGENT_ORDER):
            stats = statistics.get(agent_key, None)
            self._draw_card(col_idx, agent_key, stats)

        # Chart section
        if chart_surf is not None:
            # Centre 560px chart on 600px screen → x=20
            self.screen.blit(chart_surf, (20, 285))
        else:
            self._draw_no_data_panel()

        # Back button
        self._draw_back_button()

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    def _draw_glow_title(self, text: str, center: Tuple[int, int]) -> None:
        glow_col = (0, 160, 90)
        font = self.font_title
        passes = 3
        for r in range(passes, 0, -1):
            gs = font.render(text, True, glow_col)
            gs.set_alpha(int(35 / r))
            gx = center[0] - gs.get_width() // 2
            gy = center[1] - gs.get_height() // 2
            for dx, dy in [(-r*2,0),(r*2,0),(0,-r*2),(0,r*2),(-r,-r),(r,-r),(-r,r),(r,r)]:
                self.screen.blit(gs, (gx + dx, gy + dy))
        main = font.render(text, True, _ACCENT)
        self.screen.blit(main, main.get_rect(center=center))

    def _draw_divider(self, y: int) -> None:
        pygame.draw.line(self.screen, _BORDER, (20, y), (self.screen_width - 20, y), 1)

    # ------------------------------------------------------------------
    # Agent cards
    # ------------------------------------------------------------------

    def _draw_card(self, col_idx: int, agent_key: str, stats: Optional[Dict]) -> None:
        meta  = _AGENT_META.get(agent_key, {"label": agent_key.upper(), "color": _ACCENT, "bar": "#00ffa0"})
        cx    = _CARD_XS[col_idx]
        cy    = _CARD_Y
        cw    = _CARD_W
        ch    = _CARD_H
        hdr_h = 30
        agent_color = meta["color"]
        agent_label = meta["label"]

        card_rect = pygame.Rect(cx, cy, cw, ch)

        # Card body background
        pygame.draw.rect(self.screen, _CARD_BG, card_rect, border_radius=8)

        # Header strip (top 30px, agent colour fill with top radius only)
        hdr_rect = pygame.Rect(cx, cy, cw, hdr_h)
        pygame.draw.rect(self.screen, agent_color, hdr_rect, border_radius=8)
        # Cover bottom-radius of header with a flat rect
        pygame.draw.rect(self.screen, agent_color, (cx, cy + hdr_h - 8, cw, 8))

        # Header label
        lbl_surf = self.font_card_label.render(agent_label, True, (10, 10, 20))
        self.screen.blit(lbl_surf, lbl_surf.get_rect(center=(cx + cw // 2, cy + hdr_h // 2)))

        # Card border
        pygame.draw.rect(self.screen, agent_color, card_rect, 1, border_radius=8)

        # Stats body
        if stats and stats.get("total_games", 0) > 0:
            rows = [
                ("GAMES",     str(stats.get("total_games", 0))),
                ("AVG SCORE", f"{stats.get('avg_score', 0.0):.1f}"),
                ("MAX SCORE", str(stats.get("max_score", 0))),
                ("WIN RATE",  self._fmt_winrate(stats.get("win_rate", None))),
            ]
            row_y = cy + hdr_h + 10
            row_h = (ch - hdr_h - 14) // len(rows)
            for (lbl, val) in rows:
                lbl_s = self.font_stat_lbl.render(lbl, True, _TEXT_DIM)
                val_s = self.font_stat_val.render(val, True, _TEXT_WHITE)
                self.screen.blit(lbl_s, (cx + 10, row_y))
                self.screen.blit(val_s, (cx + 10, row_y + 13))
                row_y += row_h
        else:
            # No data
            nd_surf = self.font_nodata.render("NO DATA", True, _TEXT_DIM)
            sub_surf = self.font_nodata.render("Play a game first", True, (50, 60, 90))
            body_cy  = cy + hdr_h + (ch - hdr_h) // 2
            self.screen.blit(nd_surf, nd_surf.get_rect(center=(cx + cw // 2, body_cy - 10)))
            self.screen.blit(sub_surf, sub_surf.get_rect(center=(cx + cw // 2, body_cy + 10)))

    @staticmethod
    def _fmt_winrate(wr: Optional[float]) -> str:
        if wr is None:
            return "N/A"
        return f"{wr * 100:.1f}%"

    # ------------------------------------------------------------------
    # No-data panel (when chart cannot be generated)
    # ------------------------------------------------------------------

    def _draw_no_data_panel(self) -> None:
        panel_rect = pygame.Rect(20, 285, 560, 210)
        pygame.draw.rect(self.screen, _CARD_BG, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, _BORDER,  panel_rect, 1, border_radius=8)
        cx = 20 + 280
        cy = 285 + 105
        nd = self.font_title.render("NO DATA YET", True, _TEXT_DIM)
        sub = self.font_nodata.render("Play some games to see performance charts", True, (50, 60, 90))
        self.screen.blit(nd,  nd.get_rect(center=(cx, cy - 14)))
        self.screen.blit(sub, sub.get_rect(center=(cx, cy + 18)))

    # ------------------------------------------------------------------
    # Back button
    # ------------------------------------------------------------------

    def _draw_back_button(self) -> None:
        t = self._back_hover
        rect = self._back_button_rect()
        bg_col  = _lerp_color(_CARD_BG, (24, 28, 64), t)
        brd_col = _lerp_color(_BORDER,  _ACCENT, t)
        txt_col = _lerp_color(_TEXT_DIM, _TEXT_WHITE, t)

        pygame.draw.rect(self.screen, bg_col,  rect, border_radius=6)
        pygame.draw.rect(self.screen, brd_col, rect, 1, border_radius=6)

        lbl = self.font_back.render("\u2190 BACK TO MENU", True, txt_col)
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    # ------------------------------------------------------------------
    # Matplotlib charts
    # ------------------------------------------------------------------

    def _create_charts(self, statistics: Dict[str, Optional[Dict]]) -> Optional[pygame.Surface]:
        """Create side-by-side bar charts and return as a Pygame surface."""
        # Check if any agent has data
        has_data = any(
            statistics.get(k) and statistics[k].get("total_games", 0) > 0
            for k in _AGENT_ORDER
        )
        if not has_data:
            return None

        labels     = []
        avg_scores = []
        max_scores = []
        bar_colors = []

        for key in _AGENT_ORDER:
            meta  = _AGENT_META[key]
            stats = statistics.get(key)
            labels.append(meta["label"])
            bar_colors.append(meta["bar"])
            if stats and stats.get("total_games", 0) > 0:
                avg_scores.append(stats.get("avg_score", 0.0))
                max_scores.append(stats.get("max_score", 0))
            else:
                avg_scores.append(0)
                max_scores.append(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.0), dpi=100)
        fig.patch.set_facecolor("#06060e")

        def _style_ax(ax: plt.Axes, title: str, values: list) -> None:
            bars = ax.bar(labels, values, color=bar_colors, width=0.5)
            ax.set_title(title, color="#c8d2eb", fontsize=9, pad=4)
            ax.set_facecolor("#06060e")
            ax.tick_params(colors="#c8d2eb", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#262c64")
            # Value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + max(values) * 0.02,
                        f"{val:.1f}" if isinstance(val, float) else str(val),
                        ha="center", va="bottom",
                        color="#ffffff", fontsize=7,
                    )
            ax.set_ylim(0, max(max(values) * 1.20, 1))
            ax.yaxis.label.set_color("#c8d2eb")

        _style_ax(ax1, "AVG SCORE", avg_scores)
        _style_ax(ax2, "MAX SCORE", [float(v) for v in max_scores])

        plt.tight_layout(pad=0.8)

        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor="#06060e", edgecolor="none", dpi=100)
        buf.seek(0)
        plt.close(fig)

        chart_surf = pygame.image.load(buf, "png")
        buf.close()
        return chart_surf
