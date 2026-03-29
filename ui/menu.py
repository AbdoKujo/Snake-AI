"""Menu principal du jeu — Neural Grid cyberpunk theme."""

from __future__ import annotations
from typing import List, Tuple, Optional
from enum import Enum, auto
import math
import pygame


class MenuOption(Enum):
    """Options disponibles dans le menu."""
    PLAY_HUMAN = auto()
    PLAY_ASTAR = auto()
    PLAY_DQN = auto()
    PLAY_VERSUS = auto()
    TRAIN_DQN = auto()
    COMPARE = auto()
    STATISTICS = auto()
    QUIT = auto()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linearly interpolate between two RGB colours."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ---------------------------------------------------------------------------
# Button definitions
# ---------------------------------------------------------------------------

# Main menu buttons: (label, description, icon, accent_color, option, height)
_MAIN_BUTTONS = [
    # (label, description, icon, accent_color, option, height)
    ("PLAY",       None,                          "▶", (  0, 255, 160), None,                  54),
    ("TRAIN DQN",  "Run 1000 training episodes",  "▲", (160,  60, 200), MenuOption.TRAIN_DQN,  54),
    ("STATISTICS", "View performance comparison", "≡", (  0, 255, 160), MenuOption.STATISTICS, 54),
    ("QUIT",       "Exit the application",        "✕", (220,  60,  80), MenuOption.QUIT,        54),
]

# Play sub-menu buttons
_PLAY_BUTTONS = [
    ("← BACK",        None,                      "<", ( 75,  90, 125), None,                   52),
    ("HUMAN PLAYER",  "Arrow keys or WASD",      "▶", (255, 210,  50), MenuOption.PLAY_HUMAN,  52),
    ("A* AGENT",      "Optimal pathfinding",      "◆", ( 60, 160, 255), MenuOption.PLAY_ASTAR,  52),
    ("DQN AGENT",     "Neural network agent",     "●", (210,  80, 255), MenuOption.PLAY_DQN,    52),
    ("A* vs DQN",     "Split-screen live battle", "⚡", (130, 100, 255), MenuOption.PLAY_VERSUS, 64),
]

# Layout constants for main menu
_MAIN_BTN_X      = 40
_MAIN_BTN_W      = 520
_MAIN_BTN_H      = 54
_MAIN_BTN_GAP    = 14
_MAIN_BTN_START_Y = 210

# Layout constants for play sub-menu
_PLAY_BTN_X      = 40
_PLAY_BTN_W      = 520
_PLAY_BTN_H      = 52
_PLAY_BTN_GAP    = 10
_PLAY_BTN_START_Y = 155

# Colours
_BG         = (6, 6, 18)
_CARD_BG    = (16, 18, 44)
_CARD_HOV   = (24, 28, 64)
_BORDER     = (38, 44, 100)
_TEXT       = (200, 210, 235)
_TEXT_DIM   = (75, 90, 125)
_ACCENT     = (0, 255, 160)
_DOT_COL    = (20, 22, 50)
_NEW_BADGE  = (0, 255, 160)


class Menu:
    """Menu principal du jeu avec Neural Grid cyberpunk theme."""

    def __init__(self, screen_width: int, screen_height: int) -> None:
        self.w = screen_width
        self.h = screen_height
        self.screen_width  = screen_width
        self.screen_height = screen_height

        if not pygame.get_init():
            pygame.init()

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Snake Game AI")

        # Fonts — all Consolas
        self.font_title    = pygame.font.SysFont("consolas", 52, bold=True)
        self.font_subtitle = pygame.font.SysFont("consolas", 17, bold=False)
        self.font_label    = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_desc     = pygame.font.SysFont("consolas", 13, bold=False)
        self.font_icon     = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_arrow    = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_footer   = pygame.font.SysFont("consolas", 12, bold=False)
        self.font_new      = pygame.font.SysFont("consolas", 11, bold=True)

        # Sub-menu state
        self._show_play_submenu: bool = False
        self._fullscreen: bool = False

        # Per-button hover progress
        self._main_hover: List[float] = [0.0] * len(_MAIN_BUTTONS)
        self._play_hover: List[float] = [0.0] * len(_PLAY_BUTTONS)

        # Pre-render background with grid dots
        self._bg_surface = self._make_background()

        self.selected_option: Optional[MenuOption] = None

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def _make_background(self) -> pygame.Surface:
        """Pre-render static background with subtle grid dots."""
        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill(_BG)
        dot_spacing = 30
        for gx in range(0, self.screen_width, dot_spacing):
            for gy in range(0, self.screen_height, dot_spacing):
                pygame.draw.rect(surf, _DOT_COL, (gx, gy, 2, 2))
        return surf

    # ------------------------------------------------------------------
    # Fullscreen toggle
    # ------------------------------------------------------------------

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.w, self.h))

    # ------------------------------------------------------------------
    # Glow text
    # ------------------------------------------------------------------

    def _draw_glow_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        color: Tuple[int, int, int],
        glow_color: Tuple[int, int, int],
        center: Tuple[int, int],
        passes: int = 3,
    ) -> None:
        """Render text with a multi-pass glow halo."""
        for r in range(passes, 0, -1):
            glow_surf = font.render(text, True, glow_color)
            glow_surf.set_alpha(int(35 / r))
            gx = center[0] - glow_surf.get_width() // 2
            gy = center[1] - glow_surf.get_height() // 2
            for dx, dy in [
                (-r * 2, 0), (r * 2, 0), (0, -r * 2), (0, r * 2),
                (-r, -r),   (r, -r),   (-r, r),     (r, r),
            ]:
                surface.blit(glow_surf, (gx + dx, gy + dy))
        main = font.render(text, True, color)
        surface.blit(main, main.get_rect(center=center))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_corner_brackets(self, surface: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int], size: int = 12, thickness: int = 2) -> None:
        """Draw L-shaped corner brackets around a rect."""
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        corners = [
            # top-left
            [(x, y), (x + size, y)],
            [(x, y), (x, y + size)],
            # top-right
            [(x + w, y), (x + w - size, y)],
            [(x + w, y), (x + w, y + size)],
            # bottom-left
            [(x, y + h), (x + size, y + h)],
            [(x, y + h), (x, y + h - size)],
            # bottom-right
            [(x + w, y + h), (x + w - size, y + h)],
            [(x + w, y + h), (x + w, y + h - size)],
        ]
        for start, end in corners:
            pygame.draw.line(surface, color, start, end, thickness)

    def _draw_divider(self, surface: pygame.Surface, y: int) -> None:
        """Horizontal divider with a centre diamond."""
        cx = self.screen_width // 2
        line_color = _BORDER
        diamond_color = _ACCENT
        pygame.draw.line(surface, line_color, (40, y), (cx - 10, y), 1)
        pygame.draw.line(surface, line_color, (cx + 10, y), (self.screen_width - 40, y), 1)
        # Diamond
        diamond = [
            (cx,     y - 5),
            (cx + 5, y),
            (cx,     y + 5),
            (cx - 5, y),
        ]
        pygame.draw.polygon(surface, diamond_color, diamond)

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def run(self) -> MenuOption:
        """Display the menu and wait for the user to select an option."""
        clock = pygame.time.Clock()

        while True:
            dt = clock.tick(60) / 1000.0  # seconds
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return MenuOption.QUIT
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_click = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self._show_play_submenu:
                            self._show_play_submenu = False
                        else:
                            return MenuOption.QUIT
                    elif event.key == pygame.K_F11:
                        self._toggle_fullscreen()

            if self._show_play_submenu:
                # Update hover progress for play sub-menu buttons
                btn_y = _PLAY_BTN_START_Y
                for i, (label, desc, icon, accent, option, height) in enumerate(_PLAY_BUTTONS):
                    btn_rect = pygame.Rect(_PLAY_BTN_X, btn_y, _PLAY_BTN_W, height)
                    is_hov = btn_rect.collidepoint(mouse_pos)
                    speed = 6.0
                    if is_hov:
                        self._play_hover[i] = min(1.0, self._play_hover[i] + dt * speed)
                        if mouse_click:
                            if option is None:
                                # BACK button
                                self._show_play_submenu = False
                            else:
                                return option
                    else:
                        self._play_hover[i] = max(0.0, self._play_hover[i] - dt * speed)
                    btn_y += height + _PLAY_BTN_GAP
            else:
                # Update hover progress for main menu buttons
                btn_y = _MAIN_BTN_START_Y
                for i, (label, desc, icon, accent, option, height) in enumerate(_MAIN_BUTTONS):
                    btn_rect = pygame.Rect(_MAIN_BTN_X, btn_y, _MAIN_BTN_W, height)
                    is_hov = btn_rect.collidepoint(mouse_pos)
                    speed = 6.0
                    if is_hov:
                        self._main_hover[i] = min(1.0, self._main_hover[i] + dt * speed)
                        if mouse_click:
                            if option is None:
                                # PLAY button — show sub-menu
                                self._show_play_submenu = True
                            else:
                                return option
                    else:
                        self._main_hover[i] = max(0.0, self._main_hover[i] - dt * speed)
                    btn_y += height + _MAIN_BTN_GAP

            self._draw()

        return MenuOption.QUIT

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        """Render the full menu frame."""
        surf = self.screen

        # Background
        surf.blit(self._bg_surface, (0, 0))

        ticks = pygame.time.get_ticks()

        # --- Animated title ---
        pulse = math.sin(ticks / 1000.0 * 1.8)
        g_val = int(220 + 35 * pulse)
        title_color = (0, min(255, g_val), min(160, 130 + int(30 * pulse)))
        glow_color  = (0, 180, 100)

        title_cx = self.screen_width // 2
        title_cy = 62

        self._draw_glow_text(surf, "SNAKE GAME AI", self.font_title, title_color, glow_color, (title_cx, title_cy), passes=4)

        # Corner brackets around title area
        bracket_rect = pygame.Rect(30, 16, self.screen_width - 60, 100)
        self._draw_corner_brackets(surf, bracket_rect, _ACCENT, size=14, thickness=1)

        # Subtitle — changes based on state
        if self._show_play_submenu:
            sub_text = "[ Select Play Mode ]"
        else:
            sub_text = "[ A* vs Deep Q-Network ]"
        sub_surf = self.font_subtitle.render(sub_text, True, _TEXT_DIM)
        surf.blit(sub_surf, sub_surf.get_rect(center=(title_cx, 102)))

        # Divider
        self._draw_divider(surf, 138)

        # --- Buttons ---
        if self._show_play_submenu:
            self._draw_play_buttons(surf)
        else:
            self._draw_main_buttons(surf)

        # --- Footer ---
        left_txt  = self.font_footer.render("Snake Game AI  |  2024", True, _TEXT_DIM)
        right_txt = self.font_footer.render("F11 — FULLSCREEN  |  ESC to quit", True, _TEXT_DIM)
        footer_y  = self.screen_height - 22
        surf.blit(left_txt,  (20, footer_y))
        surf.blit(right_txt, (self.screen_width - right_txt.get_width() - 20, footer_y))

        pygame.display.flip()

    def _draw_main_buttons(self, surf: pygame.Surface) -> None:
        """Draw the main menu buttons."""
        btn_y = _MAIN_BTN_START_Y
        for i, (label, desc, icon, accent, option, height) in enumerate(_MAIN_BUTTONS):
            t = self._main_hover[i]
            btn_rect = pygame.Rect(_MAIN_BTN_X, btn_y, _MAIN_BTN_W, height)

            # Background
            bg_col = _lerp_color(_CARD_BG, _CARD_HOV, t)
            pygame.draw.rect(surf, bg_col, btn_rect, border_radius=4)

            # Left accent bar (4px)
            bar_col = _lerp_color(
                _lerp_color(_BORDER, accent, 0.4),
                accent,
                t,
            )
            bar_rect = pygame.Rect(_MAIN_BTN_X, btn_y, 4, height)
            pygame.draw.rect(surf, bar_col, bar_rect, border_radius=2)

            # Border
            border_col = _lerp_color(_BORDER, accent, t * 0.6)
            pygame.draw.rect(surf, border_col, btn_rect, 1, border_radius=4)

            # Icon
            icon_surf = self.font_icon.render(icon, True, _lerp_color(accent, (255, 255, 255), t * 0.4))
            surf.blit(icon_surf, (btn_rect.x + 18, btn_y + (height - icon_surf.get_height()) // 2))

            # Label
            lbl_col = _lerp_color(_TEXT, (255, 255, 255), t)
            lbl_surf = self.font_label.render(label, True, lbl_col)
            surf.blit(lbl_surf, (btn_rect.x + 58, btn_y + 8))

            # Description (if any)
            if desc is not None:
                desc_surf = self.font_desc.render(desc, True, _lerp_color(_TEXT_DIM, _TEXT, t * 0.7))
                surf.blit(desc_surf, (btn_rect.x + 58, btn_y + 30))
            elif option is None:
                # PLAY button — show "Select a play mode" hint
                hint_surf = self.font_desc.render("Select a play mode", True, _lerp_color(_TEXT_DIM, _TEXT, t * 0.7))
                surf.blit(hint_surf, (btn_rect.x + 58, btn_y + 30))

            # ">>" arrow on hover (right-aligned)
            if t > 0.01:
                arrow_surf = self.font_arrow.render(">>", True, accent)
                arrow_surf.set_alpha(int(255 * t))
                surf.blit(arrow_surf, (btn_rect.right - arrow_surf.get_width() - 16,
                                       btn_y + (height - arrow_surf.get_height()) // 2))

            btn_y += height + _MAIN_BTN_GAP

    def _draw_play_buttons(self, surf: pygame.Surface) -> None:
        """Draw the play sub-menu buttons."""
        btn_y = _PLAY_BTN_START_Y
        for i, (label, desc, icon, accent, option, height) in enumerate(_PLAY_BUTTONS):
            t = self._play_hover[i]
            btn_rect = pygame.Rect(_PLAY_BTN_X, btn_y, _PLAY_BTN_W, height)

            # Background
            bg_col = _lerp_color(_CARD_BG, _CARD_HOV, t)
            pygame.draw.rect(surf, bg_col, btn_rect, border_radius=4)

            # Left accent bar (4px)
            bar_col = _lerp_color(
                _lerp_color(_BORDER, accent, 0.4),
                accent,
                t,
            )
            bar_rect = pygame.Rect(_PLAY_BTN_X, btn_y, 4, height)
            pygame.draw.rect(surf, bar_col, bar_rect, border_radius=2)

            # Border
            border_col = _lerp_color(_BORDER, accent, t * 0.6)
            pygame.draw.rect(surf, border_col, btn_rect, 1, border_radius=4)

            # Icon
            icon_surf = self.font_icon.render(icon, True, _lerp_color(accent, (255, 255, 255), t * 0.4))
            surf.blit(icon_surf, (btn_rect.x + 18, btn_y + (height - icon_surf.get_height()) // 2))

            # Label
            lbl_col = _lerp_color(_TEXT, (255, 255, 255), t)
            lbl_surf = self.font_label.render(label, True, lbl_col)
            surf.blit(lbl_surf, (btn_rect.x + 58, btn_y + 8))

            # Description
            if desc is not None:
                desc_surf = self.font_desc.render(desc, True, _lerp_color(_TEXT_DIM, _TEXT, t * 0.7))
                surf.blit(desc_surf, (btn_rect.x + 58, btn_y + 30))

            # For the A* vs DQN button, add a "(NEW)" badge in bright green
            if option == MenuOption.PLAY_VERSUS:
                new_surf = self.font_new.render("(NEW)", True, _NEW_BADGE)
                surf.blit(new_surf, (btn_rect.x + 58 + (self.font_desc.render(desc or "", True, _TEXT_DIM).get_width()) + 8, btn_y + 30))

            # ">>" arrow on hover (right-aligned)
            if t > 0.01:
                arrow_surf = self.font_arrow.render(">>", True, accent)
                arrow_surf.set_alpha(int(255 * t))
                surf.blit(arrow_surf, (btn_rect.right - arrow_surf.get_width() - 16,
                                       btn_y + (height - arrow_surf.get_height()) // 2))

            btn_y += height + _PLAY_BTN_GAP
