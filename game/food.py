"""Classe Food représentant la nourriture du serpent."""

from __future__ import annotations
import random
from typing import List, Set, Optional
from .point import Point


class Food:
    """Représente la nourriture que le serpent doit manger."""
    
    def __init__(self, grid_width: int, grid_height: int) -> None:
        """
        Initialise la nourriture.
        
        Args:
            grid_width: Largeur de la grille.
            grid_height: Hauteur de la grille.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.position: Optional[Point] = None
        self.value: int = 1  # Points gagnés en mangeant
    
    def spawn(self, occupied_positions: Set[Point]) -> bool:
        """
        Place la nourriture à une position aléatoire non occupée.
        
        Args:
            occupied_positions: Ensemble des positions déjà occupées.
            
        Returns:
            True si la nourriture a pu être placée, False si la grille est pleine.
        """
        # Générer toutes les positions libres
        free_positions: List[Point] = []
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                pos = Point(x, y)
                if pos not in occupied_positions:
                    free_positions.append(pos)
        
        # Si aucune position libre, la grille est pleine (victoire!)
        if not free_positions:
            return False
        
        # Choisir une position aléatoire
        self.position = random.choice(free_positions)
        return True
    
    def get_position(self) -> Optional[Point]:
        """Retourne la position actuelle de la nourriture."""
        return self.position
    
    def is_at(self, position: Point) -> bool:
        """
        Vérifie si la nourriture est à une position donnée.
        
        Args:
            position: La position à vérifier.
            
        Returns:
            True si la nourriture est à cette position.
        """
        return self.position is not None and self.position == position
