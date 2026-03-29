"""Gestionnaire de base de données SQLite."""

from __future__ import annotations
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class DatabaseManager:
    """
    Gère toutes les interactions avec la base de données SQLite.
    Stocke les sessions de jeu et calcule les statistiques.
    """
    
    def __init__(self, db_path: str = "assets/snake_game.db") -> None:
        """
        Initialise le gestionnaire de base de données.
        
        Args:
            db_path: Chemin vers le fichier de base de données.
        """
        self.db_path = db_path
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        
        # Connexion et initialisation
        self.connection: Optional[sqlite3.Connection] = None
        self.connect()
        self._create_tables()
    
    def connect(self) -> None:
        """Établit la connexion à la base de données."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row  # Accès par nom de colonne
    
    def disconnect(self) -> None:
        """Ferme la connexion à la base de données."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def _create_tables(self) -> None:
        """Crée les tables si elles n'existent pas."""
        cursor = self.connection.cursor()
        
        # Table des sessions de jeu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS GameSession (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                final_score INTEGER NOT NULL,
                max_length INTEGER NOT NULL,
                total_moves INTEGER NOT NULL,
                duration REAL,
                game_result TEXT CHECK(game_result IN ('win', 'lose'))
            )
        """)
        
        # Table des statistiques agrégées
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT UNIQUE NOT NULL,
                total_games INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0,
                max_score INTEGER DEFAULT 0,
                min_score INTEGER,
                avg_duration REAL DEFAULT 0,
                avg_moves REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table de configuration des agents (pour DQN)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS AgentConfig (
                config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                config_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    def save_game_session(
        self,
        agent_type: str,
        final_score: int,
        max_length: int,
        total_moves: int,
        duration: float,
        game_result: str
    ) -> int:
        """
        Sauvegarde une session de jeu.
        
        Args:
            agent_type: Type d'agent ('human', 'astar', 'dqn').
            final_score: Score final.
            max_length: Longueur maximale atteinte.
            total_moves: Nombre total de mouvements.
            duration: Durée de la partie en secondes.
            game_result: Résultat ('win' ou 'lose').
            
        Returns:
            L'ID de la session créée.
        """
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO GameSession (agent_type, end_time, final_score, max_length, total_moves, duration, game_result)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        """, (agent_type, final_score, max_length, total_moves, duration, game_result))
        
        self.connection.commit()
        session_id = cursor.lastrowid
        
        # Mettre à jour les statistiques
        self._update_statistics(agent_type)
        
        return session_id
    
    def _update_statistics(self, agent_type: str) -> None:
        """
        Met à jour les statistiques agrégées pour un type d'agent.
        
        Args:
            agent_type: Le type d'agent.
        """
        cursor = self.connection.cursor()
        
        # Calculer les nouvelles statistiques
        cursor.execute("""
            SELECT 
                COUNT(*) as total_games,
                AVG(final_score) as avg_score,
                MAX(final_score) as max_score,
                MIN(final_score) as min_score,
                AVG(duration) as avg_duration,
                AVG(total_moves) as avg_moves,
                SUM(CASE WHEN game_result = 'win' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
            FROM GameSession
            WHERE agent_type = ?
        """, (agent_type,))
        
        row = cursor.fetchone()
        
        if row and row['total_games'] > 0:
            # Insérer ou mettre à jour
            cursor.execute("""
                INSERT INTO Statistics (agent_type, total_games, avg_score, max_score, min_score, avg_duration, avg_moves, win_rate, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(agent_type) DO UPDATE SET
                    total_games = excluded.total_games,
                    avg_score = excluded.avg_score,
                    max_score = excluded.max_score,
                    min_score = excluded.min_score,
                    avg_duration = excluded.avg_duration,
                    avg_moves = excluded.avg_moves,
                    win_rate = excluded.win_rate,
                    last_updated = CURRENT_TIMESTAMP
            """, (agent_type, row['total_games'], row['avg_score'], row['max_score'],
                  row['min_score'], row['avg_duration'], row['avg_moves'], row['win_rate']))
            
            self.connection.commit()
    
    def get_statistics(self, agent_type: Optional[str] = None) -> Dict[str, Dict]:
        """
        Récupère les statistiques.
        
        Args:
            agent_type: Type d'agent spécifique, ou None pour tous.
            
        Returns:
            Dictionnaire des statistiques par agent.
        """
        cursor = self.connection.cursor()
        
        if agent_type:
            cursor.execute("SELECT * FROM Statistics WHERE agent_type = ?", (agent_type,))
        else:
            cursor.execute("SELECT * FROM Statistics")
        
        rows = cursor.fetchall()
        
        stats = {}
        for row in rows:
            stats[row['agent_type']] = {
                'total_games': row['total_games'],
                'avg_score': row['avg_score'],
                'max_score': row['max_score'],
                'min_score': row['min_score'],
                'avg_duration': row['avg_duration'],
                'avg_moves': row['avg_moves'],
                'win_rate': row['win_rate'],
                'last_updated': row['last_updated'],
            }
        
        return stats
    
    def get_recent_sessions(self, agent_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Récupère les sessions de jeu récentes.
        
        Args:
            agent_type: Type d'agent (optionnel).
            limit: Nombre maximum de sessions.
            
        Returns:
            Liste des sessions.
        """
        cursor = self.connection.cursor()
        
        if agent_type:
            cursor.execute("""
                SELECT * FROM GameSession
                WHERE agent_type = ?
                ORDER BY start_time DESC
                LIMIT ?
            """, (agent_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM GameSession
                ORDER BY start_time DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_score_history(self, agent_type: str, limit: int = 100) -> List[int]:
        """
        Récupère l'historique des scores pour un agent.
        
        Args:
            agent_type: Type d'agent.
            limit: Nombre maximum de scores.
            
        Returns:
            Liste des scores (du plus ancien au plus récent).
        """
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT final_score FROM (
                SELECT final_score, start_time FROM GameSession
                WHERE agent_type = ?
                ORDER BY start_time DESC
                LIMIT ?
            ) ORDER BY start_time ASC
        """, (agent_type, limit))
        
        return [row['final_score'] for row in cursor.fetchall()]
    
    def clear_statistics(self, agent_type: Optional[str] = None) -> None:
        """
        Supprime les statistiques et sessions.
        
        Args:
            agent_type: Type d'agent (ou None pour tout supprimer).
        """
        cursor = self.connection.cursor()
        
        if agent_type:
            cursor.execute("DELETE FROM GameSession WHERE agent_type = ?", (agent_type,))
            cursor.execute("DELETE FROM Statistics WHERE agent_type = ?", (agent_type,))
        else:
            cursor.execute("DELETE FROM GameSession")
            cursor.execute("DELETE FROM Statistics")
        
        self.connection.commit()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
