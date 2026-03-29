"""Enum Direction représentant les directions de mouvement."""

from __future__ import annotations
from enum import Enum
from typing import Tuple


class Direction(Enum):
    """Énumération des directions possibles pour le serpent."""
    
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    
    def to_vector(self) -> Tuple[int, int]:
        """
        Retourne le vecteur de déplacement associé à la direction.
        
        Returns:
            Tuple (dx, dy) représentant le déplacement.
        """
        return self.value
    
    def opposite(self) -> Direction:
        """
        Retourne la direction opposée.
        
        Returns:
            La direction opposée.
        """
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]
    
    def is_opposite(self, other: Direction) -> bool:
        """
        Vérifie si une direction est l'opposée de celle-ci.
        
        Args:
            other: L'autre direction à comparer.
            
        Returns:
            True si les directions sont opposées.
        """
        return self.opposite() == other
    
    @staticmethod
    def from_index(index: int) -> Direction:
        """
        Retourne une direction à partir d'un index (0-3).
        
        Args:
            index: L'index de la direction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).
            
        Returns:
            La direction correspondante.
        """
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        return directions[index % 4]
    
    def to_index(self) -> int:
        """
        Retourne l'index de la direction (0-3).
        
        Returns:
            L'index de la direction.
        """
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        return directions.index(self)
    
    @staticmethod
    def all_directions() -> list[Direction]:
        """Retourne la liste de toutes les directions."""
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
