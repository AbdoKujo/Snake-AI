"""Training configuration screen — Neural Grid cyberpunk theme."""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import math
import pygame
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROW_H   = 54
_BTN_H   = 48
_ARROW_W = 44

_C = {
    "bg":           (6, 6, 18),
    "panel":        (12, 14, 32),
    "card":         (16, 18, 44),
    "card_device":  (14, 20, 40),
    "accent":       (0, 255, 160),
    "accent_dim":   (0, 100, 65),
    "dqn":          (210, 80, 255),
    "gpu":          (0, 200, 255),
    "text":         (200, 210, 235),
    "text_dim":     (75, 90, 125),
    "text_bright":  (255, 255, 255),
    "border":       (38, 44, 100),
    "border_gpu":   (0, 100, 160),
    "btn_bg":       (16, 18, 44),
    "btn_hover":    (24, 28, 64),
    "grid_line":    (18, 20, 48),
}


# ---------------------------------------------------------------------------
# Build param list (device options detected at runtime)
# ---------------------------------------------------------------------------

def _build_params() -> List[Dict[str, Any]]:
    import os as _os

    devices: List[str] = ["cpu"]
    device_labels: Dict[str, str] = {"cpu": "CPU  (system)"}

    if torch.cuda.is_available():
        devices.append("cuda")
        name  = torch.cuda.get_device_name(0)
        short = name if len(name) <= 22 else name[:20] + ".."
        device_labels["cuda"] = f"GPU  ({short})"

    # Worker count — auto-detect sensible default
    _cpu_count = _os.cpu_count() or 2
    _worker_values = [1, 2, 4, 6, 8]
    _default_w = max(2, min(_cpu_count - 2, 6))
    if _default_w not in _worker_values:
        _default_w = min(_worker_values, key=lambda v: abs(v - _default_w))

    return [
        {
            "label": "DEVICE",
            "key":   "device",
            "values": devices,
            "default": devices[-1],
            "fmt": lambda v, dl=device_labels: dl.get(v, v.upper()),
            "is_device": True,
        },
        {
            "label": "WORKERS",
            "key":   "num_workers",
            "values": _worker_values,
            "default": _default_w,
            "fmt": lambda v: "SERIAL" if v == 1 else f"{v} parallel",
            "is_device": False,
        },
        {
            "label": "EPISODES",
            "key":   "num_episodes",
            "values": [100, 250, 500, 1000, 2000, 5000],
            "default": 1000,
            "fmt": lambda v: f"{v:,}",
            "is_device": False,
        },
        {
            "label": "MAX STEPS / EPISODE",
            "key":   "max_steps_per_episode",
            "values": [500, 750, 1000, 1500, 2000, 3000],
            "default": 1500,
            "fmt": lambda v: f"{v:,}",
            "is_device": False,
        },
        {
            "label": "VISUALIZE EVERY",
            "key":   "visualize_every",
            "values": [0, 10, 25, 50, 100],
            "default": 50,
            "fmt": lambda v: "OFF" if v == 0 else f"every {v} ep",
            "is_device": False,
        },
        {
            "label": "TRAIN EVERY N STEPS",
            "key":   "train_freq",
            "values": [1, 2, 4, 8],
            "default": 4,
            "fmt": lambda v: f"every {v} step{'s' if v > 1 else ''}",
            "is_device": False,
        },
        {
            "label": "EPSILON DECAY",
            "key":   "epsilon_decay_type",
            "values": ["linear", "exponential"],
            "default": "exponential",
            "fmt": lambda v: v.upper(),
            "is_device": False,
        },
    ]


# ---------------------------------------------------------------------------
# View class
# ---------------------------------------------------------------------------

class TrainingConfigView:
    """
    Pygame screen that lets the user configure DQN training before launch.
    Returns a config dict from ``show()``, or None if cancelled.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width  = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("DQN Training Config — Snake AI")

        self.font_title = pygame.font.SysFont("consolas", 30, bold=True)
        self.font_label = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_value = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_btn   = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_hint  = pygame.font.SysFont("consolas", 13)
        self.clock      = pygame.time.Clock()
        self._tick      = 0

        self._params  = _build_params()
        self._indices: List[int] = []
        for p in self._params:
            try:
                self._indices.append(p["values"].index(p["default"]))
            except ValueError:
                self._indices.append(0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def show(self) -> Optional[Dict[str, Any]]:
        """Run the config screen. Returns selected config dict or None."""
        while True:
            dt = self.clock.tick(60)
            self._tick += dt
            mx, my = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_RETURN:
                        return self._build_result()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    result = self._handle_click(mx, my)
                    if result == "start":
                        return self._build_result()
                    if result == "back":
                        return None

            self._draw(mx, my)
            pygame.display.flip()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _param_rect(self, idx: int) -> pygame.Rect:
        panel_w = int(self.width * 0.72)
        panel_x = (self.width - panel_w) // 2
        top     = 128 + idx * (_ROW_H + 6)
        return pygame.Rect(panel_x, top, panel_w, _ROW_H)

    def _arrow_left_rect(self, row: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(row.right - _ARROW_W * 2 - 8,
                           row.y + (_ROW_H - _ARROW_W) // 2,
                           _ARROW_W, _ARROW_W)

    def _arrow_right_rect(self, row: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(row.right - _ARROW_W - 4,
                           row.y + (_ROW_H - _ARROW_W) // 2,
                           _ARROW_W, _ARROW_W)

    def _btn_rects(self) -> Tuple[pygame.Rect, pygame.Rect]:
        n      = len(self._params)
        base_y = 128 + n * (_ROW_H + 6) + 16
        bw, gap, cx = 200, 20, self.width // 2
        start = pygame.Rect(cx - bw - gap // 2, base_y, bw, _BTN_H)
        back  = pygame.Rect(cx + gap // 2,       base_y, bw, _BTN_H)
        return start, back

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _handle_click(self, mx: int, my: int) -> Optional[str]:
        for i, p in enumerate(self._params):
            row  = self._param_rect(i)
            if self._arrow_left_rect(row).collidepoint(mx, my):
                self._indices[i] = (self._indices[i] - 1) % len(p["values"])
            elif self._arrow_right_rect(row).collidepoint(mx, my):
                self._indices[i] = (self._indices[i] + 1) % len(p["values"])

        start, back = self._btn_rects()
        if start.collidepoint(mx, my):
            return "start"
        if back.collidepoint(mx, my):
            return "back"
        return None

    def _build_result(self) -> Dict[str, Any]:
        return {p["key"]: p["values"][self._indices[i]]
                for i, p in enumerate(self._params)}

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, mx: int, my: int) -> None:
        self.screen.fill(_C["bg"])
        self._draw_grid()
        self._draw_title()
        for i, p in enumerate(self._params):
            self._draw_param_row(i, p, mx, my)
        self._draw_buttons(mx, my)
        self._draw_footer()

    def _draw_grid(self) -> None:
        for x in range(0, self.width, 24):
            pygame.draw.line(self.screen, _C["grid_line"], (x, 0), (x, self.height))
        for y in range(0, self.height, 24):
            pygame.draw.line(self.screen, _C["grid_line"], (0, y), (self.width, y))

    def _draw_title(self) -> None:
        pulse = 0.5 + 0.5 * math.sin(self._tick / 800)
        color = tuple(int(_C["dqn"][k] + (_C["text_bright"][k] - _C["dqn"][k]) * pulse * 0.4)
                      for k in range(3))
        surf = self.font_title.render("DQN  TRAINING  CONFIG", True, color)
        self.screen.blit(surf, surf.get_rect(centerx=self.width // 2, y=38))

        sub = self.font_hint.render(
            "Use  ◀ ▶  arrows to change values  —  ENTER to start", True, _C["text_dim"])
        self.screen.blit(sub, sub.get_rect(centerx=self.width // 2, y=82))

        lw = int(self.width * 0.72)
        lx = (self.width - lw) // 2
        pygame.draw.line(self.screen, _C["accent_dim"], (lx, 112), (lx + lw, 112), 1)

    def _draw_param_row(self, idx: int, param: Dict, mx: int, my: int) -> None:
        row   = self._param_rect(idx)
        left  = self._arrow_left_rect(row)
        right = self._arrow_right_rect(row)

        val        = param["values"][self._indices[idx]]
        is_device  = param.get("is_device", False)
        is_gpu     = is_device and val == "cuda"

        # Choose accent color: GPU row uses cyan, others use green
        accent = _C["gpu"] if is_gpu else _C["accent"]
        border = _C["border_gpu"] if is_gpu else _C["border"]
        bg     = _C["card_device"] if is_device else _C["card"]

        pygame.draw.rect(self.screen, bg, row, border_radius=6)
        pygame.draw.rect(self.screen, border, row, 2 if is_device else 1, border_radius=6)

        # Label
        label_surf = self.font_label.render(param["label"], True,
                                            accent if is_device else _C["text_dim"])
        self.screen.blit(label_surf, (row.x + 16, row.y + 8))

        # Value text
        vtext    = param["fmt"](val)
        val_surf = self.font_value.render(vtext, True, accent)
        self.screen.blit(val_surf, (row.x + 16, row.y + 28))

        # GPU badge when CUDA selected
        if is_gpu:
            badge = self.font_hint.render("● CUDA ACTIVE", True, _C["gpu"])
            self.screen.blit(badge, badge.get_rect(right=left.left - 12,
                                                    centery=row.centery))

        # Arrows
        for arrow_rect, symbol in ((left, "◀"), (right, "▶")):
            hovered = arrow_rect.collidepoint(mx, my)
            pygame.draw.rect(self.screen,
                             _C["btn_hover"] if hovered else _C["btn_bg"],
                             arrow_rect, border_radius=4)
            pygame.draw.rect(self.screen, border, arrow_rect, 1, border_radius=4)
            a = self.font_value.render(symbol, True,
                                       accent if hovered else _C["text_dim"])
            self.screen.blit(a, a.get_rect(center=arrow_rect.center))

    def _draw_buttons(self, mx: int, my: int) -> None:
        start_r, back_r = self._btn_rects()

        shover = start_r.collidepoint(mx, my)
        pygame.draw.rect(self.screen, (0, 60, 40) if shover else _C["btn_bg"],
                         start_r, border_radius=6)
        pygame.draw.rect(self.screen, _C["accent"] if shover else _C["accent_dim"],
                         start_r, 2, border_radius=6)
        ss = self.font_btn.render("▶  START TRAINING", True,
                                  _C["text_bright"] if shover else _C["accent"])
        self.screen.blit(ss, ss.get_rect(center=start_r.center))

        bhover = back_r.collidepoint(mx, my)
        pygame.draw.rect(self.screen, _C["btn_hover"] if bhover else _C["btn_bg"],
                         back_r, border_radius=6)
        pygame.draw.rect(self.screen, _C["border"], back_r, 1, border_radius=6)
        bs = self.font_btn.render("← BACK", True,
                                  _C["text"] if bhover else _C["text_dim"])
        self.screen.blit(bs, bs.get_rect(center=back_r.center))

    def _draw_footer(self) -> None:
        hints = "[◀/▶]  change value       [ENTER]  start       [ESC]  back"
        surf  = self.font_hint.render(hints, True, _C["text_dim"])
        self.screen.blit(surf, surf.get_rect(centerx=self.width // 2,
                                              y=self.height - 26))
