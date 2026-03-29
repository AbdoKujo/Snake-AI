"""Classe abstraite BaseAgent pour les agents IA."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.game_state import GameState
    from game.direction import Direction


class BaseAgent(ABC):
    """
    Classe abstraite de base pour tous les agents IA.
    Définit l'interface commune que tous les agents doivent implémenter.
    """
    
    def __init__(self, name: str = "BaseAgent") -> None:
        """
        Initialise l'agent.
        
        Args:
            name: Nom de l'agent pour l'identification.
        """
        self.name = name
    
    @abstractmethod
    def get_action(self, game_state: GameState) -> Direction:
        """
        Détermine la prochaine action à effectuer.
        
        Args:
            game_state: L'état actuel du jeu.
            
        Returns:
            La direction choisie pour le prochain mouvement.
        """
        pass
    
    def train(self) -> None:
        """
        Méthode d'entraînement (optionnelle).
        Par défaut, ne fait rien. Implémentée par les agents apprenants.
        """
        pass
    
    def save_model(self, path: str) -> None:
        """
        Sauvegarde le modèle de l'agent.
        
        Args:
            path: Chemin du fichier de sauvegarde.
        """
        pass
    
    def load_model(self, path: str) -> None:
        """
        Charge le modèle de l'agent.
        
        Args:
            path: Chemin du fichier à charger.
        """
        pass
    
    def reset(self) -> None:
        """Réinitialise l'état interne de l'agent si nécessaire."""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
