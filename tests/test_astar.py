"""Tests unitaires pour l'agent A*."""

import unittest
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.point import Point
from game.direction import Direction
from game.game_state import GameState
from ai.astar_agent import AStarAgent


class TestAStarAgent(unittest.TestCase):
    """Tests pour l'agent A*."""
    
    def setUp(self):
        """Initialise l'agent et l'état du jeu pour les tests."""
        self.agent = AStarAgent()
        self.game_state = GameState(grid_width=20, grid_height=15)
    
    def test_find_path_simple(self):
        """Test la recherche de chemin simple."""
        start = Point(0, 0)
        goal = Point(5, 0)
        obstacles = set()
        
        path = self.agent.find_path(start, goal, obstacles, 20, 15)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertEqual(len(path), 6)  # 0 à 5 = 6 points
    
    def test_find_path_with_obstacles(self):
        """Test la recherche de chemin avec obstacles."""
        start = Point(0, 0)
        goal = Point(2, 0)
        obstacles = {Point(1, 0)}  # Obstacle au milieu
        
        path = self.agent.find_path(start, goal, obstacles, 20, 15)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        # Le chemin doit contourner l'obstacle
        self.assertNotIn(Point(1, 0), path)
    
    def test_find_path_no_solution(self):
        """Test quand aucun chemin n'existe."""
        start = Point(1, 1)
        goal = Point(3, 1)
        # Entourer le départ
        obstacles = {Point(0, 1), Point(2, 1), Point(1, 0), Point(1, 2)}
        
        path = self.agent.find_path(start, goal, obstacles, 20, 15)
        
        self.assertIsNone(path)
    
    def test_heuristic(self):
        """Test l'heuristique (distance de Manhattan)."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        
        h = self.agent._heuristic(p1, p2)
        
        self.assertEqual(h, 7)
    
    def test_get_action_returns_direction(self):
        """Test que get_action retourne une Direction."""
        action = self.agent.get_action(self.game_state)
        
        self.assertIsInstance(action, Direction)
        self.assertIn(action, [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
    
    def test_get_action_towards_food(self):
        """Test que l'agent se dirige vers la nourriture."""
        # Placer la nourriture à droite du serpent
        self.game_state.food.position = Point(
            self.game_state.snake.head.x + 3,
            self.game_state.snake.head.y
        )
        
        action = self.agent.get_action(self.game_state)
        
        # Devrait aller vers la droite
        self.assertEqual(action, Direction.RIGHT)


class TestAStarPathReconstruction(unittest.TestCase):
    """Tests pour la reconstruction de chemin."""
    
    def setUp(self):
        self.agent = AStarAgent()
    
    def test_reconstruct_path(self):
        """Test la reconstruction du chemin."""
        came_from = {
            Point(1, 0): Point(0, 0),
            Point(2, 0): Point(1, 0),
            Point(3, 0): Point(2, 0),
        }
        
        path = self.agent._reconstruct_path(came_from, Point(3, 0))
        
        expected = [Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)]
        self.assertEqual(path, expected)


if __name__ == '__main__':
    unittest.main()
