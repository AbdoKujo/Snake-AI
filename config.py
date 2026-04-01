"""Configuration globale du jeu Snake Game AI."""

from typing import Tuple, Dict

# =============================================================================
# PARAMÈTRES DE LA FENÊTRE
# =============================================================================
WINDOW_WIDTH: int = 720
WINDOW_HEIGHT: int = 670
FPS: int = 60

# =============================================================================
# PARAMÈTRES DE LA GRILLE DE JEU
# =============================================================================
GRID_SIZE: int = 24  # Taille d'une cellule en pixels
GRID_WIDTH: int = 30  # Nombre de cellules en largeur
GRID_HEIGHT: int = 25  # Nombre de cellules en hauteur
HUD_HEIGHT: int = 70

# =============================================================================
# COULEURS (RGB) — Neural Grid cyberpunk theme
# =============================================================================
COLORS: Dict[str, Tuple[int, int, int]] = {
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

# =============================================================================
# PARAMÈTRES DU JEU
# =============================================================================
INITIAL_SNAKE_LENGTH: int = 3
GAME_SPEED_HUMAN: int = 10  # FPS pour le mode humain
GAME_SPEED_AI: int = 30  # FPS pour le mode IA (plus rapide)
GAME_SPEED_TRAINING: int = 0  # Pas de limite pour l'entraînement
GAME_SPEED_VERSUS: int = 20
SPLIT_CELL_SIZE: int = 15

# =============================================================================
# PARAMÈTRES A*
# =============================================================================
ASTAR_VISUALIZE_PATH: bool = True  # Afficher le chemin calculé

# =============================================================================
# PARAMÈTRES DQN — CNN grid input (Phase 2)
# =============================================================================
# State: 3-channel grid (body gradient / food / free space), flattened.
# Network: Conv2d × 3 → FC trunk → Dueling heads (V + A).
# input_size = channels × GRID_HEIGHT × GRID_WIDTH = 3 × 25 × 30 = 2 250
DQN_CONFIG: Dict = {
    "learning_rate": 1e-4,           # lower LR — CNN is more sensitive than MLP
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,
    "epsilon_decay_steps": 100_000,  # more exploration needed for richer state space
    "epsilon_decay_type": "exponential",
    "batch_size": 1024,              # larger batch → GPU stays busy per call
    "target_update_freq": 1_000,     # only used when tau=0 (hard copy fallback)
    "max_memory_size": 100_000,      # ~900 MB RAM — more diverse experiences
    "train_start_size": 1_000,       # fill buffer before first gradient step
    "hidden_size": 512,              # shared FC trunk size (streams = 256 each)
    "input_size": 3 * 25 * 30,       # 2250 — flat state dim for replay buffer
    "grid_height": 25,               # must match GRID_HEIGHT
    "grid_width":  30,               # must match GRID_WIDTH
    "channels":    3,                # body / food / free-space
    "output_size": 4,
    "gradient_clip_norm": 10.0,      # CNNs tolerate higher clip than MLPs
    "double_dqn": True,
    "tau": 0.005,                    # Polyak soft target update (0 = hard copy)
    "lr_scheduler_patience": 20,     # (legacy — unused with cosine schedule)
    "lr_scheduler_factor": 0.5,      # (legacy — unused with cosine schedule)
    "lr_min": 1e-5,
    "lr_total_steps": 500_000,       # cosine LR annealing T_max
    # Prioritized Experience Replay
    "per_alpha":       0.6,
    "per_beta_start":  0.4,
    "per_beta_frames": 100_000,
    "per_epsilon":     1e-6,
}

# =============================================================================
# RÉCOMPENSES DQN
# =============================================================================
REWARD_SCHEME: Dict[str, float] = {
    # 4-signal policy — stable Q-targets, no hidden variables, no length scaling
    "eat_food":  10.0,   # flat food reward
    "die":      -10.0,   # flat death (wall or body — no distinction)
    "win":      100.0,   # fills the grid
    "approach":   0.1,   # ÷ snake_len per step (self-fading: +0.033 early → +0.001 late)
    "retreat":   -0.2,   # ÷ snake_len per step (asymmetric — oscillation always net < 0)
    # Oscillation at len=3: (+0.033 - 0.067)/2 = -0.017/step → unprofitable
    # Hard per-food step cap is set per curriculum stage (see CURRICULUM_STAGES)
    "timeout":  -10.0,   # per-food cap exceeded — same as death
}

# =============================================================================
# PARAMÈTRES D'ENTRAÎNEMENT
# =============================================================================
TRAINING_CONFIG: Dict = {
    "num_episodes":           1000,
    "max_steps_per_episode":  4500,   # GRID_WIDTH * GRID_HEIGHT * 6 — absolute ceiling
    "visualize_every":        50,     # render one episode every N; 0 = disabled
    "train_freq":             2,      # call agent.train() every N steps
}

# =============================================================================
# CURRICULUM — stage thresholds for parallel training
# Steps-per-food cap relaxes as the agent improves, mimicking progressive difficulty.
# CurriculumTrainer advances automatically when avg(last 100) crosses threshold.
# =============================================================================
CURRICULUM_STAGES: list = [
    # name                         steps_per_food        advance when avg >
    {"name": "Stage 1 — Survival",    "steps_per_food": 750,   "threshold": 5},
    {"name": "Stage 2 — Navigation",  "steps_per_food": 1500,  "threshold": 15},
    {"name": "Stage 3 — Full game",   "steps_per_food": 3000,  "threshold": 99999},
]

# =============================================================================
# PARAMÈTRES BASE DE DONNÉES
# =============================================================================
DATABASE_PATH: str = "assets/snake_game.db"

# =============================================================================
# PARAMÈTRES D'AFFICHAGE
# =============================================================================
FONT_NAME: str = "consolas"
FONT_SIZE_LARGE: int = 48
FONT_SIZE_MEDIUM: int = 32
FONT_SIZE_SMALL: int = 20
