"""Agent DQN (Deep Q-Network) pour le Snake Game."""

from __future__ import annotations
import math
import random
import json
import os
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from .neural_network import NeuralNetwork
from .replay_buffer import PrioritizedReplayBuffer

if TYPE_CHECKING:
    from game.game_state import GameState
    from game.direction import Direction


class DQNAgent(BaseAgent):
    """
    Agent utilisant Deep Q-Network pour apprendre à jouer au Snake.
    
    Implémente:
    - Experience Replay
    - Target Network (mise à jour périodique)
    - Double DQN (optionnel)
    - Epsilon-greedy avec décroissance linéaire
    """
    
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 256,
        output_size: int = 4,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        epsilon_decay_type: str = "linear",
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        max_memory_size: int = 100_000,
        train_start_size: int = 1_000,
        gradient_clip_norm: float = 1.0,
        double_dqn: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_epsilon: float = 1e-6,
        tau: float = 0.005,
        lr_scheduler_patience: int = 20,
        lr_scheduler_factor: float = 0.5,
        lr_min: float = 1e-5,
        lr_total_steps: int = 50_000,
        device: Optional[str] = None
    ) -> None:
        """
        Initialise l'agent DQN.
        
        Args:
            input_size: Dimension de l'état.
            hidden_size: Taille des couches cachées.
            output_size: Nombre d'actions possibles.
            learning_rate: Taux d'apprentissage.
            gamma: Facteur de discount.
            epsilon_start: Epsilon initial pour exploration.
            epsilon_end: Epsilon final.
            epsilon_decay_steps: Nombre de steps pour la décroissance.
            batch_size: Taille des batchs d'entraînement.
            target_update_freq: Fréquence de mise à jour du target network.
            max_memory_size: Capacité du replay buffer.
            train_start_size: Taille minimale du buffer avant entraînement.
            epsilon_decay_type: "linear" or "exponential" decay schedule.
            gradient_clip_norm: Norme maximale pour le gradient clipping.
            double_dqn: Utiliser Double DQN.
        """
        super().__init__(name="DQN")
        
        # Device — explicit override, or auto-detect
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")

        # Let cuDNN auto-tune kernel selection for fixed-size inputs
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Hyperparamètres
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_type = epsilon_decay_type
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_memory_size = max_memory_size
        self.train_start_size = train_start_size
        self.gradient_clip_norm = gradient_clip_norm
        self.double_dqn = double_dqn
        self.tau = tau
        
        # État courant
        self.epsilon = epsilon_start
        self.global_step = 0
        
        # Réseaux de neurones
        self.policy_net = NeuralNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net = NeuralNetwork(input_size, hidden_size, output_size).to(self.device)
        self.update_target_network()  # Synchroniser les poids
        self.target_net.eval()  # Target network en mode évaluation

        # Optimiseur, scheduler et fonction de perte
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=lr_total_steps,
            eta_min=lr_min,
        )
        self.loss_fn = nn.SmoothL1Loss()
        
        # Prioritized Replay Buffer — parallel numpy arrays, vectorized SumTree
        self.memory = PrioritizedReplayBuffer(
            capacity    = max_memory_size,
            state_dim   = input_size,
            alpha       = per_alpha,
            beta_start  = per_beta_start,
            beta_frames = per_beta_frames,
            per_epsilon = per_epsilon,
        )
        
        # Mixed precision (AMP) — GPU only
        self._use_amp = (self.device.type == "cuda")
        self.scaler   = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        # Statistiques d'entraînement
        self.training_losses: list[float] = []
    
    def get_action(self, game_state: GameState) -> Direction:
        """
        Choisit une action selon la stratégie epsilon-greedy.
        Calcule l'état en interne — utiliser get_action_from_state() si
        l'état est déjà disponible pour éviter le double calcul BFS.
        """
        state = game_state.get_state_representation()
        return self.get_action_from_state(state)

    def get_action_from_state(self, state: np.ndarray) -> Direction:
        """
        Choisit une action à partir d'un état déjà calculé.
        Évite le recalcul BFS quand l'état est connu dans la boucle d'entraînement.
        """
        from game.direction import Direction

        self._update_epsilon()

        if random.random() < self.epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = self.policy_net.predict(state, self.device)

        return Direction.from_index(action_idx)
    
    def remember(
        self,
        state: np.ndarray,
        action: Direction,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Stocke une transition dans le replay buffer.
        
        Args:
            state: État avant l'action.
            action: Action effectuée.
            reward: Récompense reçue.
            next_state: État après l'action.
            done: True si épisode terminé.
        """
        action_idx = action.to_index()
        self.memory.push(state, action_idx, reward, next_state, done)

    def remember_raw(self, state: np.ndarray, action_idx: int,
                     reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store a transition using raw action index (skips Direction conversion)."""
        self.memory.push(state, action_idx, reward, next_state, done)

    def train(self) -> Optional[float]:
        """
        One training step using a prioritised batch.

        Uses IS-weighted Huber loss so high-priority samples don't dominate
        gradients. Updates PER priorities with fresh TD errors after the step.

        Returns:
            Loss value, or None if the buffer isn't ready yet.
        """
        if len(self.memory) < self.train_start_size:
            return None
        if not self.memory.is_ready(self.batch_size):
            return None

        # Sample with priorities → also returns tree indices and IS weights
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        dev = self.device
        states_t      = torch.as_tensor(states,      device=dev)
        actions_t     = torch.as_tensor(actions,      device=dev)
        rewards_t     = torch.as_tensor(rewards,      device=dev)
        next_states_t = torch.as_tensor(next_states,  device=dev)
        dones_t       = torch.as_tensor(dones,        device=dev)
        weights_t     = torch.as_tensor(weights,      device=dev)

        # Q(s, a) for the actions that were actually taken
        self.policy_net.train()

        with torch.amp.autocast("cuda", enabled=self._use_amp):
            q_selected = self.policy_net(states_t) \
                             .gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Target: r + γ · Q_target(s', argmax_a Q_policy(s', a)) · (1 − done)
            with torch.no_grad():
                if self.double_dqn:
                    next_actions = self.policy_net(next_states_t).argmax(dim=1)
                    next_q_max   = self.target_net(next_states_t) \
                                       .gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_q_max = self.target_net(next_states_t).max(dim=1)[0]

                targets = rewards_t + self.gamma * next_q_max * (1 - dones_t)

            # IS-weighted element-wise Huber loss
            element_loss = F.smooth_l1_loss(q_selected, targets, reduction='none')
            loss = (weights_t * element_loss).mean()

        # TD errors for priority update (before backprop, in fp32)
        td_errors = (targets.float() - q_selected.float()).detach().cpu().numpy()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                       self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update PER priorities with fresh TD errors
        self.memory.update_priorities(indices, td_errors)

        self.global_step += 1
        self.scheduler.step()
        if self.tau > 0:
            # Soft update every training step — smooth, no instability spikes
            self.update_target_network()
        elif self.global_step % self.target_update_freq == 0:
            # Hard copy fallback when tau == 0
            self.update_target_network()

        loss_value = loss.item()
        self.training_losses.append(loss_value)
        return loss_value
    
    def update_target_network(self) -> None:
        """
        Soft Polyak update when tau > 0 (default):
            θ_target = τ·θ_policy + (1−τ)·θ_target
        Hard copy when tau == 0 (legacy).
        """
        if self.tau > 0:
            for p_tgt, p_src in zip(self.target_net.parameters(),
                                    self.policy_net.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p_src.data)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_scheduler(self, avg_score: float) -> None:
        """No-op — LR is auto-stepped via cosine annealing in train()."""
        pass
    
    def _update_epsilon(self) -> None:
        """Met à jour epsilon selon le schedule choisi (linéaire ou exponentiel)."""
        if self.global_step >= self.epsilon_decay_steps:
            self.epsilon = self.epsilon_end
        elif self.epsilon_decay_type == "exponential":
            # Decay rate so that epsilon reaches epsilon_end at epsilon_decay_steps
            decay_rate = math.log(self.epsilon_end / self.epsilon_start) / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start * math.exp(decay_rate * self.global_step)
        else:  # linear
            decay_ratio = self.global_step / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_ratio
    
    def save_model(self, path: str) -> None:
        """
        Sauvegarde le modèle et les hyperparamètres.

        Args:
            path: Chemin du fichier de sauvegarde (.pt).
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        checkpoint = {
            'policy_net_state_dict':  self.policy_net.state_dict(),
            'target_net_state_dict':  self.target_net.state_dict(),
            'optimizer_state_dict':   self.optimizer.state_dict(),
            'scheduler_state_dict':   self.scheduler.state_dict(),
            'epsilon':     self.epsilon,
            'global_step': self.global_step,
            'hyperparameters': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'double_dqn': self.double_dqn,
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Charge le modèle et les hyperparamètres.
        
        Args:
            path: Chemin du fichier à charger (.pt).
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (KeyError, ValueError, TypeError):
                pass  # incompatible scheduler type — use fresh cosine schedule
        self.epsilon     = checkpoint['epsilon']
        self.global_step = checkpoint['global_step']
        
        print(f"Model loaded from {path}")
        print(f"Epsilon: {self.epsilon:.4f}, Global step: {self.global_step}")
    
    def set_eval_mode(self) -> None:
        """Met l'agent en mode évaluation (epsilon = 0)."""
        self.epsilon = 0.0
        self.policy_net.eval()
    
    def set_train_mode(self) -> None:
        """Remet l'agent en mode entraînement."""
        self._update_epsilon()
        self.policy_net.train()
    
    def get_q_values(self, game_state: GameState) -> np.ndarray:
        """
        Retourne les Q-values pour l'état actuel (pour visualisation).
        
        Args:
            game_state: L'état du jeu.
            
        Returns:
            Array des Q-values pour chaque action.
        """
        state = game_state.get_state_representation()
        return self.policy_net.get_q_values(state, self.device)
    
    def save_buffer(self, path: str) -> None:
        """Persist the replay buffer alongside the model checkpoint."""
        self.memory.save(path)
        print(f"  [Buffer] Saved {len(self.memory):,} transitions → {path}")

    def load_buffer(self, path: str) -> bool:
        """Restore the replay buffer from a previous run.  Returns True on success."""
        ok = self.memory.load(path)
        if ok:
            print(f"  [Buffer] Loaded {len(self.memory):,} transitions ← {path}")
        return ok

    def get_stats(self) -> Dict:
        """Retourne les statistiques d'entraînement."""
        return {
            'epsilon':    self.epsilon,
            'global_step': self.global_step,
            'memory_size': len(self.memory),
            'avg_loss':   np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'lr':         self.optimizer.param_groups[0]['lr'],
        }
