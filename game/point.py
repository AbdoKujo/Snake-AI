"""Classe Point représentant une position 2D sur la grille."""

from __future__ import annotations
from typing import Tuple
import math


class Point:
    """Représente un point (x, y) sur la grille de jeu."""
    
    __slots__ = ("x", "y")
    
    def __init__(self, x: int, y: int) -> None:
        """
        Initialise un point avec des coordonnées.
        
        Args:
            x: Coordonnée horizontale.
            y: Coordonnée verticale.
        """
        self.x = x
        self.y = y
    
    def __eq__(self, other: object) -> bool:
        """Vérifie l'égalité entre deux points."""
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        """Retourne le hash du point (pour utilisation dans sets/dicts)."""
        return hash((self.x, self.y))
    
    def __repr__(self) -> str:
        """Représentation string du point."""
        return f"Point({self.x}, {self.y})"
    
    def __add__(self, other: Point) -> Point:
        """Addition de deux points."""
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Point) -> Point:
        """Soustraction de deux points."""
        return Point(self.x - other.x, self.y - other.y)
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convertit le point en tuple."""
        return (self.x, self.y)
    
    def distance_to(self, other: Point) -> float:
        """
        Calcule la distance euclidienne vers un autre point.
        
        Args:
            other: Le point cible.
            
        Returns:
            La distance euclidienne.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def manhattan_distance(self, other: Point) -> int:
        """
        Calcule la distance de Manhattan vers un autre point.
        
        Args:
            other: Le point cible.
            
        Returns:
            La distance de Manhattan.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def copy(self) -> Point:
        """Retourne une copie du point."""
        return Point(self.x, self.y)
