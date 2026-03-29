"""Tests unitaires pour l'agent DQN."""

import unittest
import sys
import os
import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.point import Point
from game.direction import Direction
from game.game_state import GameState
from ai.dqn_agent import DQNAgent
from ai.replay_buffer import ReplayBuffer
from ai.neural_network import NeuralNetwork


class TestReplayBuffer(unittest.TestCase):
    """Tests pour le ReplayBuffer."""
    
    def setUp(self):
        """Initialise le buffer pour les tests."""
        self.buffer = ReplayBuffer(capacity=100)
    
    def test_push(self):
        """Test l'ajout de transitions."""
        state = np.zeros(11, dtype=np.float32)
        next_state = np.ones(11, dtype=np.float32)
        
        self.buffer.push(state, 0, 1.0, next_state, False)
        
        self.assertEqual(len(self.buffer), 1)
    
    def test_sample(self):
        """Test l'échantillonnage."""
        # Ajouter plusieurs transitions
        for i in range(20):
            state = np.full(11, i, dtype=np.float32)
            next_state = np.full(11, i + 1, dtype=np.float32)
            self.buffer.push(state, i % 4, float(i), next_state, i == 19)
        
        states, actions, rewards, next_states, dones = self.buffer.sample(5)
        
        self.assertEqual(states.shape, (5, 11))
        self.assertEqual(actions.shape, (5,))
        self.assertEqual(rewards.shape, (5,))
        self.assertEqual(next_states.shape, (5, 11))
        self.assertEqual(dones.shape, (5,))
    
    def test_is_ready(self):
        """Test la vérification de disponibilité."""
        self.assertFalse(self.buffer.is_ready(10))
        
        # Ajouter assez de transitions
        for i in range(15):
            self.buffer.push(np.zeros(11), 0, 0.0, np.zeros(11), False)
        
        self.assertTrue(self.buffer.is_ready(10))
    
    def test_capacity_limit(self):
        """Test la limite de capacité."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(20):
            buffer.push(np.zeros(11), 0, 0.0, np.zeros(11), False)
        
        self.assertEqual(len(buffer), 10)


class TestNeuralNetwork(unittest.TestCase):
    """Tests pour le réseau de neurones."""
    
    def setUp(self):
        """Initialise le réseau pour les tests."""
        self.network = NeuralNetwork(input_size=11, hidden_size=64, output_size=4)
    
    def test_forward_shape(self):
        """Test la forme de sortie du forward pass."""
        import torch
        
        batch_size = 5
        x = torch.randn(batch_size, 11)
        output = self.network(x)
        
        self.assertEqual(output.shape, (batch_size, 4))
    
    def test_predict(self):
        """Test la prédiction."""
        import torch
        
        state = np.random.randn(11).astype(np.float32)
        device = torch.device("cpu")
        
        action = self.network.predict(state, device)
        
        self.assertIsInstance(action, int)
        self.assertIn(action, [0, 1, 2, 3])


class TestDQNAgent(unittest.TestCase):
    """Tests pour l'agent DQN."""
    
    def setUp(self):
        """Initialise l'agent pour les tests."""
        self.agent = DQNAgent(
            input_size=11,
            hidden_size=64,
            output_size=4,
            batch_size=8,
            train_start_size=10,
            max_memory_size=100
        )
        self.game_state = GameState(grid_width=20, grid_height=15)
    
    def test_get_action_returns_direction(self):
        """Test que get_action retourne une Direction."""
        action = self.agent.get_action(self.game_state)
        
        self.assertIsInstance(action, Direction)
    
    def test_remember(self):
        """Test la mémorisation des transitions."""
        state = self.game_state.get_state_representation()
        action = Direction.RIGHT
        reward = 1.0
        next_state = state.copy()
        done = False
        
        initial_size = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), initial_size + 1)
    
    def test_train_not_ready(self):
        """Test que train ne fait rien si pas assez de données."""
        result = self.agent.train()
        
        self.assertIsNone(result)
    
    def test_train_ready(self):
        """Test l'entraînement avec assez de données."""
        # Remplir le buffer
        state = self.game_state.get_state_representation()
        for _ in range(20):
            self.agent.remember(state, Direction.RIGHT, 1.0, state, False)
        
        result = self.agent.train()
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, float)
    
    def test_epsilon_decay(self):
        """Test la décroissance d'epsilon."""
        initial_epsilon = self.agent.epsilon
        
        # Simuler plusieurs steps
        for _ in range(100):
            self.agent.global_step += 1
            self.agent._update_epsilon()
        
        self.assertLess(self.agent.epsilon, initial_epsilon)
    
    def test_update_target_network(self):
        """Test la mise à jour du target network."""
        import torch
        
        # Modifier les poids du policy network
        with torch.no_grad():
            self.agent.policy_net.fc1.weight.fill_(1.0)
        
        # Les poids devraient être différents
        policy_weight = self.agent.policy_net.fc1.weight[0, 0].item()
        target_weight = self.agent.target_net.fc1.weight[0, 0].item()
        self.assertNotEqual(policy_weight, target_weight)
        
        # Mettre à jour
        self.agent.update_target_network()
        
        # Les poids devraient être identiques
        target_weight = self.agent.target_net.fc1.weight[0, 0].item()
        self.assertEqual(policy_weight, target_weight)
    
    def test_get_stats(self):
        """Test la récupération des statistiques."""
        stats = self.agent.get_stats()
        
        self.assertIn('epsilon', stats)
        self.assertIn('global_step', stats)
        self.assertIn('memory_size', stats)


class TestGameStateRepresentation(unittest.TestCase):
    """Tests pour la représentation d'état."""
    
    def setUp(self):
        """Initialise l'état du jeu pour les tests."""
        self.game_state = GameState(grid_width=20, grid_height=15)
    
    def test_state_representation_shape(self):
        """Test la forme de la représentation d'état."""
        state = self.game_state.get_state_representation()
        
        self.assertEqual(state.shape, (11,))
        self.assertEqual(state.dtype, np.float32)
    
    def test_state_representation_values(self):
        """Test les valeurs de la représentation d'état."""
        state = self.game_state.get_state_representation()
        
        # Toutes les valeurs devraient être 0 ou 1
        for value in state:
            self.assertIn(value, [0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
