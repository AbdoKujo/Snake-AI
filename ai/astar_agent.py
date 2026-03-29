"""Agent A* pour le Snake Game."""

from __future__ import annotations
import heapq
from typing import Dict, List, Set, Tuple, Optional, TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from game.game_state import GameState
    from game.point import Point
    from game.direction import Direction


class AStarAgent(BaseAgent):
    """
    Agent utilisant l'algorithme A* pour trouver le chemin optimal
    vers la nourriture en évitant les obstacles.
    """
    
    def __init__(self) -> None:
        """Initialise l'agent A*."""
        super().__init__(name="A*")
        self.current_path: List[Point] = []
    
    def get_action(self, game_state: GameState) -> Direction:
        """
        Détermine la prochaine direction en utilisant A*.
        
        Args:
            game_state: L'état actuel du jeu.
            
        Returns:
            La direction optimale vers la nourriture.
        """
        from game.direction import Direction
        from game.point import Point
        
        head = game_state.snake.head
        food = game_state.food.position
        
        # Calculer le chemin vers la nourriture
        path = self.find_path(
            start=head,
            goal=food,
            obstacles=game_state.snake.get_body_set(),
            grid_width=game_state.grid_width,
            grid_height=game_state.grid_height
        )
        
        if path and len(path) > 1:
            # Le chemin inclut le point de départ, prendre le suivant
            next_pos = path[1]
            self.current_path = path
            
            # Convertir en direction
            dx = next_pos.x - head.x
            dy = next_pos.y - head.y
            
            if dx == 1:
                return Direction.RIGHT
            elif dx == -1:
                return Direction.LEFT
            elif dy == 1:
                return Direction.DOWN
            elif dy == -1:
                return Direction.UP
        
        # Si pas de chemin, essayer de survivre
        return self._get_survival_move(game_state)
    
    def find_path(
        self,
        start: Point,
        goal: Point,
        obstacles: Set[Point],
        grid_width: int,
        grid_height: int
    ) -> Optional[List[Point]]:
        """
        Trouve le chemin le plus court entre start et goal en évitant les obstacles.
        
        Args:
            start: Point de départ.
            goal: Point d'arrivée.
            obstacles: Ensemble des points à éviter.
            grid_width: Largeur de la grille.
            grid_height: Hauteur de la grille.
            
        Returns:
            Liste des points formant le chemin, ou None si aucun chemin.
        """
        from game.point import Point
        
        # File de priorité: (f_score, counter, point)
        # counter pour départager les égalités de f_score
        counter = 0
        open_set: List[Tuple[int, int, Point]] = []
        heapq.heappush(open_set, (0, counter, start))
        
        # Pour reconstruire le chemin
        came_from: Dict[Point, Point] = {}
        
        # g_score: coût du chemin depuis le départ
        g_score: Dict[Point, float] = {start: 0}
        
        # f_score: g_score + heuristique
        f_score: Dict[Point, float] = {start: self._heuristic(start, goal)}
        
        # Points déjà explorés
        closed_set: Set[Point] = set()
        
        while open_set:
            # Extraire le point avec le plus petit f_score
            _, _, current = heapq.heappop(open_set)
            
            # Vérifier si on a atteint l'objectif
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Explorer les voisins
            for neighbor in self._get_neighbors(current, obstacles, grid_width, grid_height):
                if neighbor in closed_set:
                    continue
                
                # Coût pour atteindre ce voisin
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Meilleur chemin trouvé
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # Aucun chemin trouvé
        return None
    
    def _heuristic(self, a: Point, b: Point) -> int:
        """
        Calcule l'heuristique (distance de Manhattan).
        
        Args:
            a: Premier point.
            b: Second point.
            
        Returns:
            Distance de Manhattan entre les deux points.
        """
        return abs(a.x - b.x) + abs(a.y - b.y)
    
    def _reconstruct_path(self, came_from: Dict[Point, Point], current: Point) -> List[Point]:
        """
        Reconstruit le chemin à partir du dictionnaire came_from.
        
        Args:
            came_from: Dictionnaire des prédécesseurs.
            current: Point final.
            
        Returns:
            Liste des points formant le chemin (du début à la fin).
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _get_neighbors(
        self,
        point: Point,
        obstacles: Set[Point],
        grid_width: int,
        grid_height: int
    ) -> List[Point]:
        """
        Retourne les voisins valides d'un point.
        
        Args:
            point: Le point central.
            obstacles: Points à éviter.
            grid_width: Largeur de la grille.
            grid_height: Hauteur de la grille.
            
        Returns:
            Liste des voisins accessibles.
        """
        from game.point import Point
        
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        for dx, dy in directions:
            nx, ny = point.x + dx, point.y + dy
            neighbor = Point(nx, ny)
            
            # Vérifier les limites
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                # Vérifier les obstacles (corps du serpent)
                # Note: On exclut le dernier segment car il va bouger
                if neighbor not in obstacles:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _get_survival_move(self, game_state: GameState) -> Direction:
        """
        Quand aucun chemin vers la nourriture n'existe,
        choisir un mouvement qui maximise la survie.
        
        Args:
            game_state: L'état actuel du jeu.
            
        Returns:
            La direction la plus sûre.
        """
        from game.direction import Direction
        
        head = game_state.snake.head
        current_direction = game_state.snake.direction
        
        # Essayer d'abord de continuer tout droit
        valid_moves = game_state.get_valid_moves()
        
        if not valid_moves:
            # Aucun mouvement valide, retourner la direction actuelle
            return current_direction
        
        # Préférer continuer dans la même direction si possible
        if current_direction in valid_moves:
            return current_direction
        
        # Sinon, choisir le mouvement qui donne le plus d'espace
        best_move = valid_moves[0]
        best_space = 0
        
        for move in valid_moves:
            space = self._count_reachable_cells(game_state, move)
            if space > best_space:
                best_space = space
                best_move = move
        
        return best_move
    
    def _count_reachable_cells(self, game_state: GameState, direction: Direction) -> int:
        """
        Compte le nombre de cellules accessibles après un mouvement.
        
        Args:
            game_state: L'état actuel du jeu.
            direction: La direction du mouvement.
            
        Returns:
            Nombre de cellules accessibles.
        """
        from game.point import Point
        
        # Simuler le mouvement
        dx, dy = direction.to_vector()
        new_head = Point(game_state.snake.head.x + dx, game_state.snake.head.y + dy)
        
        # BFS pour compter les cellules accessibles
        obstacles = game_state.snake.get_body_set()
        visited = {new_head}
        queue = [new_head]
        count = 1
        
        while queue:
            current = queue.pop(0)
            for nx, ny in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = Point(current.x + nx, current.y + ny)
                
                if (0 <= neighbor.x < game_state.grid_width and
                    0 <= neighbor.y < game_state.grid_height and
                    neighbor not in obstacles and
                    neighbor not in visited):
                    visited.add(neighbor)
                    queue.append(neighbor)
                    count += 1
        
        return count
    
    def get_current_path(self) -> List[Point]:
        """Retourne le chemin actuel pour l'affichage."""
        return self.current_path
