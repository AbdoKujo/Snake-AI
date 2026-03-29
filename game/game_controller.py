"""Classe GameController - Chef d'orchestre du jeu."""

from __future__ import annotations
import pygame
from typing import Optional, TYPE_CHECKING
from enum import Enum, auto

from .game_state import GameState
from .direction import Direction

if TYPE_CHECKING:
    from ai.base_agent import BaseAgent
    from ui.renderer import Renderer
    from data.database_manager import DatabaseManager


class GameMode(Enum):
    """Modes de jeu disponibles."""
    HUMAN = auto()
    ASTAR = auto()
    DQN = auto()
    DQN_TRAINING = auto()
    COMPARISON = auto()


class GameController:
    """
    Contrôleur principal du jeu.
    Orchestre les interactions entre le modèle (GameState), 
    la vue (Renderer) et l'IA (BaseAgent).
    """
    
    def __init__(
        self,
        game_state: GameState,
        renderer: Optional[Renderer] = None,
        agent: Optional[BaseAgent] = None,
        db_manager: Optional[DatabaseManager] = None,
        mode: GameMode = GameMode.HUMAN,
        game_speed: int = 10
    ) -> None:
        """
        Initialise le contrôleur de jeu.
        
        Args:
            game_state: L'état du jeu à contrôler.
            renderer: Le renderer pour l'affichage (optionnel pour l'entraînement).
            agent: L'agent IA (optionnel pour le mode humain).
            db_manager: Le gestionnaire de base de données.
            mode: Le mode de jeu.
            game_speed: Vitesse du jeu en FPS (0 = pas de limite).
        """
        self.game_state = game_state
        self.renderer = renderer
        self.agent = agent
        self.db_manager = db_manager
        self.mode = mode
        self.game_speed = game_speed
        
        self.running: bool = False
        self.paused: bool = False
        self.clock = pygame.time.Clock()
        
        # Pour le mode entraînement
        self.episode_count: int = 0
        self.total_reward: float = 0.0
    
    def run(self) -> None:
        """Boucle principale du jeu."""
        self.running = True
        
        while self.running:
            # Gestion des événements
            self.handle_events()
            
            if not self.paused and not self.game_state.game_over:
                # Mise à jour du jeu
                self.update()
            
            # Rendu (si renderer disponible)
            if self.renderer:
                self.renderer.render(self.game_state, self.agent)
            
            # Limiter le FPS
            if self.game_speed > 0:
                self.clock.tick(self.game_speed)
        
        # Sauvegarder la session si nécessaire
        self._save_session()
    
    def run_training_episode(self, max_steps: int = 0, train_freq: int = 1) -> Tuple[int, float]:
        """
        Exécute un épisode d'entraînement sans rendu.

        Args:
            max_steps:  Maximum steps before forcing episode end (0 = no limit).
            train_freq: Call agent.train() every N steps (default 1 = every step).
                        Setting to 4 reduces training overhead ~4× with no quality loss.

        Returns:
            Tuple (score, total_reward).
        """
        self.game_state.reset()
        self.total_reward = 0.0
        done  = False
        steps = 0

        # Compute initial state once — reused each iteration to avoid double BFS
        state = self.game_state.get_state_representation()

        has_remember = hasattr(self.agent, 'remember')
        has_train    = hasattr(self.agent, 'train')
        get_action   = getattr(self.agent, 'get_action_from_state',
                               lambda s: self.agent.get_action(self.game_state))

        while not done:
            action = get_action(state)

            done, reward = self.game_state.update(action)
            steps += 1

            # Timeout — same cost as death: looping is a complete failure
            if max_steps > 0 and steps >= max_steps and not done:
                reward += -10.0
                done = True

            next_state = self.game_state.get_state_representation()

            if has_remember:
                self.agent.remember(state, action, reward, next_state, done)

            if has_train and steps % train_freq == 0:
                self.agent.train()

            state = next_state
            self.total_reward += reward

        self.episode_count += 1
        return self.game_state.score, self.total_reward
    
    def handle_events(self) -> None:
        """Gère les événements Pygame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
    
    def _handle_keydown(self, key: int) -> None:
        """
        Gère les touches pressées.
        
        Args:
            key: Code de la touche pressée.
        """
        # Touches universelles
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_p or key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_F11:
            if self.renderer:
                self.renderer.toggle_fullscreen()

        # Contrôles du serpent (mode humain uniquement)
        if self.mode == GameMode.HUMAN:
            direction_map = {
                pygame.K_UP: Direction.UP,
                pygame.K_DOWN: Direction.DOWN,
                pygame.K_LEFT: Direction.LEFT,
                pygame.K_RIGHT: Direction.RIGHT,
                pygame.K_w: Direction.UP,
                pygame.K_s: Direction.DOWN,
                pygame.K_a: Direction.LEFT,
                pygame.K_d: Direction.RIGHT,
            }
            
            if key in direction_map:
                self.game_state.snake.change_direction(direction_map[key])
    
    def update(self) -> None:
        """Met à jour l'état du jeu."""
        if self.mode == GameMode.HUMAN:
            # Mode humain: utiliser la direction actuelle du serpent
            self.game_state.update()
        
        elif self.agent is not None:
            # Modes IA: demander l'action à l'agent
            action = self.agent.get_action(self.game_state)
            done, reward = self.game_state.update(action)
            
            # Pour l'entraînement DQN
            if self.mode == GameMode.DQN_TRAINING:
                self.total_reward += reward
                if done:
                    self.episode_count += 1
    
    def reset(self) -> None:
        """Réinitialise le jeu."""
        self.game_state.reset()
        self.total_reward = 0.0
    
    def pause(self) -> None:
        """Met le jeu en pause ou le reprend."""
        self.paused = not self.paused
    
    def stop(self) -> None:
        """Arrête la boucle de jeu."""
        self.running = False
    
    def _save_session(self) -> None:
        """Sauvegarde la session de jeu dans la base de données."""
        if self.db_manager is None:
            return
        
        agent_type = "human"
        if self.mode == GameMode.ASTAR:
            agent_type = "astar"
        elif self.mode in (GameMode.DQN, GameMode.DQN_TRAINING):
            agent_type = "dqn"
        
        result = "win" if self.game_state.won else "lose"
        
        self.db_manager.save_game_session(
            agent_type=agent_type,
            final_score=self.game_state.score,
            max_length=self.game_state.snake.length,
            total_moves=self.game_state.moves,
            duration=self.game_state.get_elapsed_time(),
            game_result=result
        )


# Import pour le type hint de Tuple
from typing import Tuple
