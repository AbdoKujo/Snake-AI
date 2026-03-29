"""Tests unitaires pour le module Snake."""

import unittest
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.point import Point
from game.direction import Direction
from game.snake import Snake


class TestPoint(unittest.TestCase):
    """Tests pour la classe Point."""
    
    def test_creation(self):
        """Test la création d'un point."""
        p = Point(5, 10)
        self.assertEqual(p.x, 5)
        self.assertEqual(p.y, 10)
    
    def test_equality(self):
        """Test l'égalité entre points."""
        p1 = Point(3, 4)
        p2 = Point(3, 4)
        p3 = Point(3, 5)
        
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
    
    def test_hash(self):
        """Test le hash pour utilisation dans set/dict."""
        p1 = Point(3, 4)
        p2 = Point(3, 4)
        
        points = {p1}
        self.assertIn(p2, points)
    
    def test_manhattan_distance(self):
        """Test la distance de Manhattan."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        
        self.assertEqual(p1.manhattan_distance(p2), 7)
        self.assertEqual(p2.manhattan_distance(p1), 7)
    
    def test_addition(self):
        """Test l'addition de points."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        result = p1 + p2
        
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 6)


class TestDirection(unittest.TestCase):
    """Tests pour l'enum Direction."""
    
    def test_opposite(self):
        """Test les directions opposées."""
        self.assertEqual(Direction.UP.opposite(), Direction.DOWN)
        self.assertEqual(Direction.DOWN.opposite(), Direction.UP)
        self.assertEqual(Direction.LEFT.opposite(), Direction.RIGHT)
        self.assertEqual(Direction.RIGHT.opposite(), Direction.LEFT)
    
    def test_is_opposite(self):
        """Test la vérification d'opposition."""
        self.assertTrue(Direction.UP.is_opposite(Direction.DOWN))
        self.assertFalse(Direction.UP.is_opposite(Direction.LEFT))
    
    def test_to_vector(self):
        """Test la conversion en vecteur."""
        self.assertEqual(Direction.UP.to_vector(), (0, -1))
        self.assertEqual(Direction.DOWN.to_vector(), (0, 1))
        self.assertEqual(Direction.LEFT.to_vector(), (-1, 0))
        self.assertEqual(Direction.RIGHT.to_vector(), (1, 0))
    
    def test_from_index(self):
        """Test la conversion depuis un index."""
        self.assertEqual(Direction.from_index(0), Direction.UP)
        self.assertEqual(Direction.from_index(1), Direction.DOWN)
        self.assertEqual(Direction.from_index(2), Direction.LEFT)
        self.assertEqual(Direction.from_index(3), Direction.RIGHT)


class TestSnake(unittest.TestCase):
    """Tests pour la classe Snake."""
    
    def setUp(self):
        """Initialise un serpent pour les tests."""
        self.snake = Snake(Point(5, 5), initial_length=3)
    
    def test_initial_length(self):
        """Test la longueur initiale."""
        self.assertEqual(self.snake.length, 3)
    
    def test_initial_position(self):
        """Test la position initiale."""
        self.assertEqual(self.snake.head.x, 5)
        self.assertEqual(self.snake.head.y, 5)
    
    def test_initial_direction(self):
        """Test la direction initiale."""
        self.assertEqual(self.snake.direction, Direction.RIGHT)
    
    def test_move(self):
        """Test le mouvement."""
        old_head = Point(self.snake.head.x, self.snake.head.y)
        self.snake.move()
        
        # La tête devrait être à droite
        self.assertEqual(self.snake.head.x, old_head.x + 1)
        self.assertEqual(self.snake.head.y, old_head.y)
        
        # La longueur devrait rester la même
        self.assertEqual(self.snake.length, 3)
    
    def test_grow(self):
        """Test la croissance."""
        self.snake.grow()
        self.snake.move()
        
        # La longueur devrait augmenter
        self.assertEqual(self.snake.length, 4)
    
    def test_change_direction_valid(self):
        """Test le changement de direction valide."""
        # Peut tourner à gauche
        result = self.snake.change_direction(Direction.UP)
        self.assertTrue(result)
        self.assertEqual(self.snake.direction, Direction.UP)
    
    def test_change_direction_invalid(self):
        """Test le changement de direction invalide (demi-tour)."""
        # Ne peut pas faire demi-tour
        result = self.snake.change_direction(Direction.LEFT)
        self.assertFalse(result)
        self.assertEqual(self.snake.direction, Direction.RIGHT)
    
    def test_self_collision(self):
        """Test la collision avec soi-même."""
        # Serpent initial ne se collisionne pas
        self.assertFalse(self.snake.check_self_collision())
        
        # Créer une situation de collision
        self.snake.body = [Point(5, 5), Point(4, 5), Point(4, 4), Point(5, 4), Point(5, 5)]
        self.assertTrue(self.snake.check_self_collision())
    
    def test_wall_collision(self):
        """Test la collision avec les murs."""
        grid_width = 10
        grid_height = 10
        
        # Position normale
        self.assertFalse(self.snake.check_wall_collision(grid_width, grid_height))
        
        # Serpent hors limites
        self.snake.body[0] = Point(-1, 5)
        self.assertTrue(self.snake.check_wall_collision(grid_width, grid_height))


if __name__ == '__main__':
    unittest.main()
