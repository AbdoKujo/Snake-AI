"""Classe Snake représentant le serpent du jeu."""

from __future__ import annotations
from typing import List, Set
from .point import Point
from .direction import Direction


class Snake:
    """Représente le serpent contrôlé par le joueur ou l'IA."""
    
    def __init__(self, start_pos: Point, initial_length: int = 3) -> None:
        """
        Initialise le serpent.
        
        Args:
            start_pos: Position initiale de la tête.
            initial_length: Longueur initiale du serpent.
        """
        self.body: List[Point] = []
        self.direction: Direction = Direction.RIGHT
        self.growing: bool = False
        self.alive: bool = True
        
        # Créer le corps initial (horizontal vers la gauche)
        for i in range(initial_length):
            self.body.append(Point(start_pos.x - i, start_pos.y))
    
    @property
    def head(self) -> Point:
        """Retourne la position de la tête du serpent."""
        return self.body[0]
    
    @property
    def tail(self) -> List[Point]:
        """Retourne le corps sans la tête."""
        return self.body[1:]
    
    @property
    def length(self) -> int:
        """Retourne la longueur du serpent."""
        return len(self.body)
    
    def move(self) -> None:
        """
        Déplace le serpent d'une case dans la direction actuelle.
        La queue suit le corps sauf si le serpent est en train de grandir.
        """
        if not self.alive:
            return
        
        # Calculer la nouvelle position de la tête
        dx, dy = self.direction.to_vector()
        new_head = Point(self.head.x + dx, self.head.y + dy)
        
        # Insérer la nouvelle tête au début
        self.body.insert(0, new_head)
        
        # Supprimer la queue sauf si on grandit
        if self.growing:
            self.growing = False
        else:
            self.body.pop()
    
    def grow(self) -> None:
        """Marque le serpent pour grandir au prochain mouvement."""
        self.growing = True
    
    def change_direction(self, new_direction: Direction) -> bool:
        """
        Change la direction du serpent si ce n'est pas un demi-tour.
        
        Args:
            new_direction: La nouvelle direction souhaitée.
            
        Returns:
            True si le changement a été effectué, False sinon.
        """
        # Empêcher le demi-tour (collision immédiate avec soi-même)
        if self.direction.is_opposite(new_direction):
            return False
        
        self.direction = new_direction
        return True
    
    def check_self_collision(self) -> bool:
        """
        Vérifie si la tête est en collision avec le corps.
        
        Returns:
            True si collision, False sinon.
        """
        return self.head in self.tail
    
    def check_wall_collision(self, grid_width: int, grid_height: int) -> bool:
        """
        Vérifie si la tête est hors des limites de la grille.
        
        Args:
            grid_width: Largeur de la grille.
            grid_height: Hauteur de la grille.
            
        Returns:
            True si hors limites, False sinon.
        """
        return (
            self.head.x < 0 or
            self.head.x >= grid_width or
            self.head.y < 0 or
            self.head.y >= grid_height
        )
    
    def check_collision(self, grid_width: int, grid_height: int) -> bool:
        """
        Vérifie toutes les collisions (murs et soi-même).
        
        Args:
            grid_width: Largeur de la grille.
            grid_height: Hauteur de la grille.
            
        Returns:
            True si collision, False sinon.
        """
        return (
            self.check_wall_collision(grid_width, grid_height) or
            self.check_self_collision()
        )
    
    def get_body_positions(self) -> List[Point]:
        """Retourne la liste des positions occupées par le serpent."""
        return self.body.copy()
    
    def get_body_set(self) -> Set[Point]:
        """Retourne un ensemble des positions occupées (pour recherche rapide)."""
        return set(self.body)
    
    def will_collide_at(self, position: Point, grid_width: int, grid_height: int) -> bool:
        """
        Vérifie si une position donnée causerait une collision.
        
        Args:
            position: La position à vérifier.
            grid_width: Largeur de la grille.
            grid_height: Hauteur de la grille.
            
        Returns:
            True si collision potentielle, False sinon.
        """
        # Vérifier les murs
        if (position.x < 0 or position.x >= grid_width or
            position.y < 0 or position.y >= grid_height):
            return True
        
        # Vérifier le corps (sauf la queue qui va bouger)
        body_without_tail = self.body[:-1]
        return position in body_without_tail
    
    def reset(self, start_pos: Point, initial_length: int = 3) -> None:
        """
        Réinitialise le serpent à son état initial.
        
        Args:
            start_pos: Nouvelle position de départ.
            initial_length: Longueur initiale.
        """
        self.body.clear()
        for i in range(initial_length):
            self.body.append(Point(start_pos.x - i, start_pos.y))
        self.direction = Direction.RIGHT
        self.growing = False
        self.alive = True
