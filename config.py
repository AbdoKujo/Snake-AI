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
# PARAMÈTRES DQN
# =============================================================================
DQN_CONFIG: Dict = {
    "learning_rate": 1e-3,
    "gamma": 0.99,                   # effective horizon ~100 steps — stable for Snake
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,             # less noise in late training
    "epsilon_decay_steps": 30_000,
    "epsilon_decay_type": "exponential",
    "batch_size": 512,
    "target_update_freq": 1_000,     # only used when tau=0 (hard copy fallback)
    "max_memory_size": 100_000,
    "train_start_size": 20_000,
    "hidden_size": 512,              # wider streams for Dueling (256 each)
    "input_size": 14,
    "output_size": 4,
    "gradient_clip_norm": 1.0,
    "double_dqn": True,
    "tau": 0.005,                    # Polyak soft target update (0 = hard copy)
    "lr_scheduler_patience": 20,     # (legacy — unused with cosine schedule)
    "lr_scheduler_factor": 0.5,      # (legacy — unused with cosine schedule)
    "lr_min": 1e-5,
    "lr_total_steps": 500_000,       # cosine LR annealing T_max
    # Prioritized Experience Replay
    "per_alpha":       0.6,          # 0 = uniform, 1 = fully prioritised
    "per_beta_start":  0.4,          # IS correction start (anneals to 1.0)
    "per_beta_frames": 100_000,      # steps to reach β = 1.0
    "per_epsilon":     1e-6,         # small constant to avoid zero priority
}

# =============================================================================
# RÉCOMPENSES DQN
# =============================================================================
REWARD_SCHEME: Dict[str, float] = {
    "eat_food":       10.0,
    "milestone":       5.0,   # bonus every 10 lengths grown (gradient signal)
    "die_base":      -10.0,   # base death penalty
    "die_per_cell":   -0.1,   # extra per body cell — scales with progress lost
    "win":           100.0,
    "step_factor":    -0.03,  # divided by snake_length → short=-0.01, long→0
    "timeout":       -10.0,   # same as death — looping is total failure
}

# =============================================================================
# PARAMÈTRES D'ENTRAÎNEMENT
# =============================================================================
TRAINING_CONFIG: Dict = {
    "num_episodes":           1000,
    "max_steps_per_episode":  1500,   # GRID_WIDTH * GRID_HEIGHT * 2 — prevents loops
    "visualize_every":        50,     # render one episode every N; 0 = disabled
    "train_freq":             2,      # call agent.train() every N steps
}

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
