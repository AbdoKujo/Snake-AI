"""Vectorized DQN trainer — N game environments in a single process.

Why this is faster than the parallel (multiprocessing) trainer
──────────────────────────────────────────────────────────────
Parallel trainer bottleneck:
  8 workers → pickle(23 MB state dict) → Queue → drain loop → train(batch=256)
  GPU busy: ~5-10%   (waiting for queue I/O between every tiny batch)

Vectorized trainer:
  N envs step sequentially (no IPC, no pickle, no queue)
  All N states → ONE GPU forward pass → N actions
  Collect N experiences → train(batch=1024) once
  GPU busy: ~70-85%

The key insight: GPU overhead is ~5 ms per call regardless of batch size.
  256  samples: 5 ms overhead + 1 ms compute =  6 ms  (17% efficiency)
  1024 samples: 5 ms overhead + 3 ms compute =  8 ms  (38% efficiency)
  4096 samples: 5 ms overhead + 8 ms compute = 13 ms  (62% efficiency)
Bigger batches = GPU actually does useful work per call.

Epsilon spread (Ape-X style)
─────────────────────────────
Env 0  gets eps_min → mostly exploits (greedy play)
Env N  gets eps_max → mostly explores (random play)
Intermediate envs spaced geometrically for diverse experience.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Ape-X epsilon spread
# ---------------------------------------------------------------------------

def _env_epsilons(n_envs: int,
                  eps_min: float = 0.01,
                  eps_max: float = 1.0) -> np.ndarray:
    """Geometrically spaced epsilons from eps_min (env 0) to eps_max (env N-1)."""
    if n_envs == 1:
        return np.array([0.1], dtype=np.float32)
    ratio = eps_max / eps_min
    return np.array(
        [eps_min * ratio ** (i / (n_envs - 1)) for i in range(n_envs)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# VectorizedTrainer
# ---------------------------------------------------------------------------

class VectorizedTrainer:
    """
    Runs N Snake game instances in a single process.

    Each training step:
      1. Batch GPU inference  — one forward pass for all N states
      2. Step all N envs      — pure Python, no IPC
      3. Store N experiences  — direct buffer push, no queue
      4. Train on GPU         — large batch (1024+) every train_freq steps

    Compatible with CurriculumTrainer: exposes the same .train() signature
    as ParallelTrainer so CurriculumTrainer needs zero changes.
    """

    def __init__(
        self,
        agent,
        n_envs:      int   = 16,
        grid_width:  int   = 30,
        grid_height: int   = 25,
        eps_max:     float = 1.0,
    ) -> None:
        self.agent       = agent
        self.n_envs      = n_envs
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.eps_max     = eps_max

    # ------------------------------------------------------------------
    def train(
        self,
        num_episodes:      int  = 1000,
        max_steps:         int  = 4500,
        steps_per_food:    int  = 0,       # 0 = disabled (curriculum sets this)
        train_freq:        int  = 4,       # train once per N total env-steps
        log_interval:      int  = 10,
        save_path:         str  = "saved_models/dqn_model.pt",
        checkpoint_every:  int  = 100,
        weight_sync_every: int  = 0,       # unused (no workers) — kept for API compat
        visualize_every:   int  = 0,
        visualize_fn:      Optional[Callable] = None,
        on_log:            Optional[Callable] = None,
        on_episode:        Optional[Callable] = None,  # callback(score) every episode
    ) -> Dict[str, Any]:
        """
        Run vectorized training.

        Args:
            num_episodes:    total episodes to complete across all envs.
            max_steps:       hard per-episode step cap.
            steps_per_food:  per-food step cap (0 = disabled). Set by curriculum.
            train_freq:      call agent.train() every N total env-steps.
            log_interval:    log stats every N completed episodes.
            save_path:       model checkpoint path.
            checkpoint_every:save checkpoint every N episodes.
            visualize_every: render one episode every N episodes (0 = off).
            visualize_fn:    callback(agent) for visual episodes.
            on_log:          callback(ep, total, score, avg, best, stats, elapsed, eta).
            on_episode:      lightweight callback(score) fired for every completed episode.

        Returns:
            Dict: episodes, best_score, avg_score, total_steps, train_steps, elapsed.
        """
        from game.game_state import GameState
        from game.direction import Direction

        agent  = self.agent
        N      = self.n_envs
        gw, gh = self.grid_width, self.grid_height

        # Epsilon spread (Ape-X style):
        #   eps_min = agent.epsilon_end  (always one near-greedy env)
        #   eps_max = decays with agent.global_step (less random over time)
        # Spread is recomputed after every training step so it tracks the
        # agent's actual learning progress.
        eps_min = agent.epsilon_end   # e.g. 0.02

        def _current_epsilons() -> np.ndarray:
            """Recompute spread using the agent's current decayed epsilon as max."""
            agent._update_epsilon()                          # sync agent.epsilon
            cur_max = max(agent.epsilon, eps_min + 0.01)    # keep spread > 0
            return _env_epsilons(N, eps_min=eps_min, eps_max=cur_max)

        epsilons = _current_epsilons()
        print(f"  Env ε spread: [{epsilons[0]:.3f} … {epsilons[-1]:.3f}]  "
              f"({N} envs, no IPC)")

        # ── Initialise N environments ──────────────────────────────────
        envs: List[GameState] = [GameState(gw, gh) for _ in range(N)]

        # Pre-allocate state arrays — avoids per-step allocations
        input_size  = agent.input_size
        states      = np.array([e.get_state_representation() for e in envs],
                               dtype=np.float32)
        next_states = np.empty((N, input_size), dtype=np.float32)

        # Per-env counters
        steps_per_env        = np.zeros(N, dtype=np.int32)
        steps_since_food_env = np.zeros(N, dtype=np.int32)
        prev_scores_env      = np.zeros(N, dtype=np.int32)

        # Global counters
        episodes_done = 0
        total_steps   = 0
        train_steps   = 0
        best_score    = 0
        scores: List[int] = []
        start_time    = time.time()

        def _fmt(s: float) -> str:
            s = int(s)
            h, r = divmod(s, 3600)
            m, sc = divmod(r, 60)
            return f"{h}h{m:02d}m{sc:02d}s" if h else f"{m}m{sc:02d}s"

        # ── Main training loop ─────────────────────────────────────────
        try:
            while episodes_done < num_episodes:

                # ── 1. Batch GPU inference (ONE forward pass for all N envs) ──
                action_indices = agent.get_actions_batch(states, epsilons)

                # ── 2. Step every env & collect experiences ────────────────
                for i in range(N):
                    action = Direction.from_index(int(action_indices[i]))
                    done, reward = envs[i].update(action)

                    steps_per_env[i]        += 1
                    steps_since_food_env[i] += 1
                    total_steps             += 1

                    # Detect food eaten → reset per-food counter
                    cur_score = envs[i].score
                    if cur_score > prev_scores_env[i]:
                        prev_scores_env[i]      = cur_score
                        steps_since_food_env[i] = 0

                    # Hard per-food cap (curriculum stage pressure)
                    if (steps_per_food > 0
                            and steps_since_food_env[i] >= steps_per_food
                            and not done):
                        reward = -10.0
                        done   = True

                    # Hard per-episode cap
                    if steps_per_env[i] >= max_steps and not done:
                        reward += -10.0
                        done    = True

                    # Compute next state
                    next_states[i] = envs[i].get_state_representation()

                    # Push experience directly — no queue, no pickle
                    agent.remember_raw(
                        states[i], int(action_indices[i]),
                        reward, next_states[i], float(done),
                    )

                    # ── Episode end ──────────────────────────────────────
                    if done:
                        ep_score = envs[i].score
                        episodes_done += 1
                        scores.append(ep_score)
                        if on_episode:
                            on_episode(ep_score)

                        if ep_score > best_score:
                            best_score = ep_score
                            os.makedirs(
                                os.path.dirname(save_path) or ".", exist_ok=True)
                            agent.save_model(save_path)

                        if episodes_done % log_interval == 0:
                            recent  = scores[-100:]
                            avg     = sum(recent) / len(recent)
                            elapsed = time.time() - start_time
                            rate    = episodes_done / max(elapsed, 1e-9)
                            eta     = (num_episodes - episodes_done) / max(rate, 1e-9)
                            stats   = agent.get_stats()
                            if on_log:
                                on_log(episodes_done, num_episodes,
                                       ep_score, avg, best_score,
                                       stats, elapsed, eta)

                        if episodes_done % checkpoint_every == 0:
                            os.makedirs(
                                os.path.dirname(save_path) or ".", exist_ok=True)
                            agent.save_model(save_path)
                            print(f"  ↳ Checkpoint saved at episode {episodes_done}")

                        if (visualize_every > 0 and visualize_fn
                                and episodes_done % visualize_every == 0):
                            visualize_fn(agent)

                        # Reset this env
                        envs[i].reset()
                        next_states[i]           = envs[i].get_state_representation()
                        steps_per_env[i]         = 0
                        steps_since_food_env[i]  = 0
                        prev_scores_env[i]       = 0

                        if episodes_done >= num_episodes:
                            break   # exit inner loop

                # Roll states forward
                states[:] = next_states

                # ── 3. Train once per outer iteration (after N env steps) ──
                # Firing once here instead of N/train_freq times inside the
                # loop keeps the Python overhead proportional to env steps,
                # not multiplied by N.  Effective cadence: one gradient step
                # per train_freq outer iterations = train_freq × N env steps.
                if total_steps % (train_freq * N) < N:
                    loss = agent.train()
                    if loss is not None:
                        train_steps += 1
                        epsilons = _current_epsilons()

        except KeyboardInterrupt:
            # Return partial results — CurriculumTrainer will propagate aborted flag
            elapsed   = time.time() - start_time
            avg_score = sum(scores[-100:]) / max(len(scores[-100:]), 1)
            return {
                "episodes":    episodes_done,
                "best_score":  best_score,
                "avg_score":   avg_score,
                "total_steps": total_steps,
                "train_steps": train_steps,
                "elapsed":     elapsed,
                "aborted":     True,
            }

        elapsed = time.time() - start_time
        avg_score = sum(scores[-100:]) / max(len(scores[-100:]), 1)

        return {
            "episodes":    episodes_done,
            "best_score":  best_score,
            "avg_score":   avg_score,
            "total_steps": total_steps,
            "train_steps": train_steps,
            "elapsed":     elapsed,
        }
