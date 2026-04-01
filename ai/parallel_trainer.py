"""Parallel DQN training — N worker processes generating experiences for one GPU trainer."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue as _queue
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Message tags for the experience queue
# ---------------------------------------------------------------------------
_TAG_EXP  = 0   # (0, state, action_idx, reward, next_state, done_f)
_TAG_DONE = 1   # (1, worker_id, score, steps)


# ---------------------------------------------------------------------------
# Per-worker epsilon (Ape-X style)
# ---------------------------------------------------------------------------

def _worker_epsilons(num_workers: int,
                     eps_min: float = 0.01,
                     eps_max: float = 1.0) -> List[float]:
    """Log-uniform spread of epsilons across workers.

    Worker 0 gets *eps_min* (mostly exploit), worker N-1 gets *eps_max*
    (mostly explore).  Intermediate workers are spaced geometrically.
    """
    if num_workers == 1:
        return [0.1]
    ratio = eps_max / eps_min
    return [round(eps_min * ratio ** (i / (num_workers - 1)), 4)
            for i in range(num_workers)]


# ---------------------------------------------------------------------------
# Worker process (runs on a separate CPU core — no GPU, no pygame)
# ---------------------------------------------------------------------------

def _env_worker(
    worker_id: int,
    exp_queue:   mp.Queue,
    weight_recv: mp.Queue,      # Queue(maxsize=1) — non-blocking weight updates
    done_queue:  mp.Queue,      # unbounded — DONE messages must never be dropped
    stop_event:  mp.Event,
    config: Dict[str, Any],
) -> None:
    """Generate experiences in a loop, pushing them to *exp_queue*."""
    # Suppress pygame welcome banner that leaks through game.__init__
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

    # Lazy imports — avoids pulling in pygame display initialisation
    from game.game_state import GameState
    from game.direction import Direction
    from ai.neural_network import NeuralNetwork

    gw              = config["grid_width"]
    gh              = config["grid_height"]
    max_steps       = config["max_steps"]
    steps_per_food  = config.get("steps_per_food", 0)   # 0 = disabled

    game = GameState(gw, gh)
    net  = NeuralNetwork(
        config["input_size"], config["hidden_size"], config["output_size"],
        grid_height=config["grid_height"],
        grid_width =config["grid_width"],
        channels   =config["channels"],
    )
    net.eval()
    cpu = torch.device("cpu")

    # Fixed per-worker epsilon (Ape-X style — no decay)
    epsilon: float = config["worker_epsilon"]

    # Load initial weights sent by the main process
    sd = config.get("initial_state_dict")
    if sd is not None:
        net.load_state_dict(sd)

    while not stop_event.is_set():
        # --- pull weight updates (non-blocking) ----------------------------
        # Queue(maxsize=1): at most one pending update; drain all available.
        # Never blocks — main process uses put_nowait which skips if worker
        # hasn't consumed the previous update yet.
        while True:
            try:
                msg = weight_recv.get_nowait()
                # msg is pre-pickled bytes (serialised once in main process)
                # → workers only pay the unpickle cost, not the pickle cost
                sd = pickle.loads(msg) if isinstance(msg, (bytes, bytearray)) else msg
                net.load_state_dict(sd)
            except _queue.Empty:
                break
            except Exception:
                break

        # --- run one episode -----------------------------------------------
        game.reset()
        state           = game.get_state_representation()
        done            = False
        steps           = 0
        steps_since_food = 0
        prev_score      = 0

        while not done and not stop_event.is_set():
            # epsilon-greedy action selection (CPU only)
            if random.random() < epsilon:
                action_idx = random.randint(0, config["output_size"] - 1)
            else:
                with torch.no_grad():
                    action_idx = net.predict(state, cpu)

            action = Direction.from_index(action_idx)
            done, reward = game.update(action)
            steps            += 1
            steps_since_food += 1

            # Detect food eaten by score increase → reset per-food counter
            if game.score > prev_score:
                prev_score       = game.score
                steps_since_food = 0

            # Hard per-food step cap (curriculum stage timeout)
            if steps_per_food > 0 and steps_since_food >= steps_per_food and not done:
                reward = -10.0
                done   = True

            # Hard per-episode step cap
            if max_steps > 0 and steps >= max_steps and not done:
                reward += -10.0
                done = True

            next_state = game.get_state_representation()

            try:
                exp_queue.put(
                    (_TAG_EXP, state, action_idx, reward, next_state, float(done)),
                    timeout=2.0,
                )
            except _queue.Full:
                pass  # drop experience under back-pressure — count is not critical

            state = next_state

        # --- report episode completion (must never be dropped) -------------
        # done_queue is unbounded — put() here never blocks or raises Full.
        if not stop_event.is_set():
            done_queue.put((_TAG_DONE, worker_id, game.score, steps))


# ---------------------------------------------------------------------------
# ParallelTrainer
# ---------------------------------------------------------------------------

class ParallelTrainer:
    """
    N CPU workers → shared experience queue → 1 GPU trainer.

    Workers run game episodes (including the expensive BFS state computation)
    in parallel on separate cores.  The main process pulls experiences off
    the queue, feeds them to the replay buffer, and trains on GPU.

    IPC design:
      - exp_queue  (maxsize=20_000): experience transitions; drops OK under back-pressure.
      - done_queue (unbounded):      episode-completion signals; MUST never be dropped.
      - weight_queues[i] (maxsize=1): per-worker weight broadcasts; non-blocking put_nowait.
    """

    def __init__(
        self,
        agent,                          # DQNAgent (lives in main process)
        num_workers: int  = 0,          # 0 → auto-detect
        grid_width:  int  = 30,
        grid_height: int  = 25,
        eps_max:     float = 1.0,       # upper end of worker epsilon spread
    ) -> None:
        self.agent       = agent
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.eps_max     = eps_max
        self.num_workers = (
            num_workers if num_workers > 0
            else max(1, min((os.cpu_count() or 2) - 2, 6))
        )

    # ------------------------------------------------------------------
    def train(
        self,
        num_episodes:      int  = 1000,
        max_steps:         int  = 1500,
        steps_per_food:    int  = 0,        # 0 = disabled; curriculum sets this per stage
        train_freq:        int  = 2,
        log_interval:      int  = 10,
        save_path:         str  = "saved_models/dqn_model.pt",
        checkpoint_every:  int  = 100,
        weight_sync_every: int  = 200,
        visualize_every:   int  = 0,
        visualize_fn:      Optional[Callable] = None,
        on_log:            Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run parallel training.

        Args:
            num_episodes:      total episodes to run across all workers.
            max_steps:         max steps per episode before timeout.
            train_freq:        call agent.train() every N experiences received.
            log_interval:      print stats every N episodes.
            save_path:         where to save checkpoint .pt files.
            checkpoint_every:  save every N episodes.
            weight_sync_every: broadcast weights to workers every N train steps.
                               Non-blocking (put_nowait) so high values just mean
                               less-fresh worker weights — never causes stalls.
            visualize_every:   run a visual episode every N episodes (0 = off).
            visualize_fn:      callback(agent) to render one episode in main process.
            on_log:            callback(ep, total, score, avg, best, stats, elapsed, eta).

        Returns:
            Dict with keys: episodes, best_score, avg_score, total_steps,
            train_steps, elapsed.
        """
        agent = self.agent
        nw    = self.num_workers

        # -- initial weights for workers (CPU tensors, pickle-safe) --------
        sd_cpu = {k: v.cpu().clone() for k, v in agent.policy_net.state_dict().items()}

        # Per-worker epsilon spread (Ape-X style)
        epsilons = _worker_epsilons(nw, eps_max=self.eps_max)
        print(f"  Worker ε: [{', '.join(f'{e:.3f}' for e in epsilons)}]")

        # Set agent epsilon to the best-worker value for meaningful stats display.
        # (agent.epsilon is never updated in parallel mode since get_action is
        # not called in the main process — this makes logged epsilon meaningful.)
        agent.epsilon = min(epsilons)

        worker_base_cfg: Dict[str, Any] = {
            "grid_width":         self.grid_width,
            "grid_height":        self.grid_height,
            "input_size":         agent.input_size,
            "hidden_size":        agent.hidden_size,
            "output_size":        agent.output_size,
            "channels":           agent.channels,
            "max_steps":          max_steps,
            "steps_per_food":     steps_per_food,
            "initial_state_dict": sd_cpu,
        }

        # -- IPC channels ---------------------------------------------------
        # exp_queue:     bounded — experiences may be dropped under back-pressure.
        # done_queue:    unbounded — episode completion signals MUST never be dropped.
        # weight_queues: one Queue(maxsize=1) per worker — non-blocking weight push.
        exp_queue     = mp.Queue(maxsize=20_000)
        done_queue    = mp.Queue()               # no maxsize — never drops
        stop_event    = mp.Event()               # single shared event for all workers
        weight_queues: List[mp.Queue] = []

        # -- spawn workers -------------------------------------------------
        workers: List[mp.Process] = []

        for wid in range(nw):
            wq   = mp.Queue(maxsize=1)           # at most one pending weight update
            wcfg = {**worker_base_cfg, "worker_epsilon": epsilons[wid]}
            weight_queues.append(wq)
            p = mp.Process(
                target=_env_worker,
                args=(wid, exp_queue, wq, done_queue, stop_event, wcfg),
                daemon=True,
            )
            workers.append(p)

        for p in workers:
            p.start()

        # -- main training loop --------------------------------------------
        episodes_done     = 0
        total_steps       = 0
        train_steps       = 0
        best_score        = 0
        scores: List[int] = []
        start_time        = time.time()
        last_heartbeat    = time.time()
        last_done_time    = time.time()   # watchdog: time of last DONE received
        next_sync         = weight_sync_every

        # Local helper — process one _TAG_DONE item and update all shared state.
        # Called from both the Phase-0 drain and the post-exp-drain.
        # Returns True when episodes_done has reached the target.
        def _process_done(score: int) -> bool:
            nonlocal episodes_done, best_score, last_done_time, last_heartbeat
            episodes_done  += 1
            last_done_time  = time.time()
            scores.append(score)

            if score > best_score:
                best_score = score
                agent.save_model(save_path)

            if episodes_done % log_interval == 0:
                recent  = scores[-100:]
                avg_sc  = sum(recent) / len(recent)
                elapsed = time.time() - start_time
                eta     = (elapsed / episodes_done) * (num_episodes - episodes_done)
                stats   = agent.get_stats()
                if on_log:
                    on_log(episodes_done, num_episodes, score,
                           avg_sc, best_score, stats, elapsed, eta)
                agent.update_scheduler(avg_sc)
                last_heartbeat = time.time()

            if episodes_done % checkpoint_every == 0:
                agent.save_model(save_path)
                print(f"  ↳ Checkpoint saved at episode {episodes_done}")

            if (visualize_every > 0 and visualize_fn
                    and episodes_done % visualize_every == 0):
                visualize_fn(agent)

            return episodes_done >= num_episodes

        _DRAIN_BATCH       = 10_000   # max experiences to drain per cycle
        _SLOW_CYCLE        = 5.0      # warn if cycle exceeds this (seconds)
        _WATCHDOG_WARN     = 120.0    # seconds before printing stall warning
        _WATCHDOG_ABORT    = 300.0    # seconds before aborting the loop
        _CHECK_ALIVE_EVERY = 50       # check worker liveness every N cycles
        _cycle_count       = 0

        try:
            while episodes_done < num_episodes:
                cycle_t0 = time.time()
                _cycle_count += 1

                # ==========================================================
                # Phase 0 — DONE QUEUE: drain episode-completion signals
                # Always non-blocking; must run before exp drain every cycle.
                # ==========================================================
                while True:
                    try:
                        item = done_queue.get_nowait()
                    except _queue.Empty:
                        break
                    _, _wid, score, _steps = item
                    if _process_done(score):
                        break   # reached num_episodes

                if episodes_done >= num_episodes:
                    break

                # ----------------------------------------------------------
                # Watchdog: detect stall (BUG-3)
                # Only check after the DONE drain above — if we received any
                # DONE this cycle, last_done_time was just updated.
                # ----------------------------------------------------------
                stall_secs = time.time() - last_done_time
                if stall_secs > _WATCHDOG_WARN:
                    alive = sum(1 for p in workers if p.is_alive())
                    print(f"  WARNING: No episode completed for {stall_secs:.0f}s "
                          f"({alive}/{nw} workers alive, "
                          f"{episodes_done}/{num_episodes} done)")
                    if stall_secs > _WATCHDOG_ABORT:
                        print(f"  ABORTING: stall exceeded {_WATCHDOG_ABORT:.0f}s — "
                              f"breaking out of training loop.")
                        break

                # ----------------------------------------------------------
                # Periodic worker aliveness check (BUG-8)
                # ----------------------------------------------------------
                if _cycle_count % _CHECK_ALIVE_EVERY == 0:
                    alive_count = sum(1 for p in workers if p.is_alive())
                    if alive_count < nw:
                        print(f"  WARNING: {nw - alive_count} worker(s) have died "
                              f"({alive_count}/{nw} alive, "
                              f"{episodes_done}/{num_episodes} episodes done)")
                    if alive_count == 0:
                        print("  All workers exited unexpectedly — stopping.")
                        break

                # ==========================================================
                # Phase 1 — DRAIN: pull up to _DRAIN_BATCH experiences
                # ==========================================================
                new_exps = 0
                drained  = 0

                while drained < _DRAIN_BATCH:
                    try:
                        if drained == 0:
                            item = exp_queue.get(timeout=2.0)
                        else:
                            item = exp_queue.get_nowait()
                    except _queue.Empty:
                        break

                    drained += 1
                    tag = item[0]

                    if tag == _TAG_EXP:
                        _, state, action_idx, reward, next_state, done_f = item
                        agent.remember_raw(state, action_idx, reward, next_state, bool(done_f))
                        total_steps += 1
                        new_exps    += 1
                    # _TAG_DONE items should no longer appear in exp_queue,
                    # but ignore gracefully if they do (legacy or edge case).

                drain_t = time.time() - cycle_t0

                # Drain done_queue again after waiting on exp_queue — DONE messages
                # that arrived during the exp drain are processed immediately rather
                # than waiting for the next outer iteration.
                while True:
                    try:
                        item = done_queue.get_nowait()
                    except _queue.Empty:
                        break
                    _, _wid, score, _steps = item
                    if _process_done(score):
                        break

                if episodes_done >= num_episodes:
                    break

                # Check if exp_queue was completely empty and all workers dead
                if drained == 0:
                    if not any(p.is_alive() for p in workers):
                        print("  All workers exited unexpectedly — stopping.")
                        break
                    continue

                # ==========================================================
                # Phase 2 — TRAIN: batch training proportional to drained
                # ==========================================================
                train_t0 = time.time()
                num_trains_done = 0
                if new_exps > 0:
                    # Hard cap per cycle: keeps each cycle ≤ ~500 ms so the
                    # exp_queue never saturates and workers never block.
                    # At 60 ms/step, 8 steps = 480 ms → workers generate
                    # ~1500 new experiences in that window → queue stays empty.
                    # (Old buffer_capacity cap grew unbounded as the buffer
                    # filled, causing 7-second cycles that flooded the queue.)
                    num_trains = min(max(1, new_exps // train_freq), 8)
                    for _ in range(num_trains):
                        loss = agent.train()
                        if loss is not None:
                            train_steps     += 1
                            num_trains_done += 1
                train_t = time.time() - train_t0

                # ==========================================================
                # Phase 3 — SYNC: broadcast weights to workers (non-blocking)
                # Uses Queue(maxsize=1) per worker — put_nowait skips if the
                # worker hasn't consumed the previous update yet.
                # No more blocking pc.send() — eliminates the 32s stall (BUG-2/6).
                # ==========================================================
                sync_t0 = time.time()
                if train_steps >= next_sync:
                    sd = {k: v.cpu().clone()
                          for k, v in agent.policy_net.state_dict().items()}
                    # Pickle ONCE — send the same bytes object to every worker.
                    # With a 23 MB CNN state dict, this saves 8× pickle cost vs
                    # calling put_nowait(sd) separately (which pickles per call).
                    sd_bytes = pickle.dumps(sd)
                    for wq in weight_queues:
                        try:
                            wq.put_nowait(sd_bytes)
                        except _queue.Full:
                            pass  # worker hasn't read last update; next sync will push fresh weights
                    next_sync = train_steps + weight_sync_every
                sync_t = time.time() - sync_t0

                # Warn about slow cycles so we can diagnose stalls
                cycle_t = time.time() - cycle_t0
                if cycle_t > _SLOW_CYCLE:
                    print(f"  ⚠ Slow cycle {cycle_t:.1f}s — "
                          f"drain: {drain_t:.1f}s ({drained} items) | "
                          f"train: {train_t:.1f}s ({num_trains_done} steps) | "
                          f"sync: {sync_t:.1f}s")

                # ==========================================================
                # Heartbeat: show progress even between episode completions
                # ==========================================================
                now = time.time()
                if now - last_heartbeat > 30.0:
                    elapsed = now - start_time
                    qsize   = exp_queue.qsize() if hasattr(exp_queue, 'qsize') else -1
                    dqsize  = done_queue.qsize() if hasattr(done_queue, 'qsize') else -1
                    mem_sz  = len(agent.memory)
                    lr = agent.optimizer.param_groups[0]['lr']
                    print(f"  ♦ {episodes_done}/{num_episodes} ep | "
                          f"{total_steps:,} steps | {train_steps:,} trained | "
                          f"exp_q: {qsize} | done_q: {dqsize} | buffer: {mem_sz:,} | "
                          f"lr: {lr:.2e} | "
                          f"elapsed: {int(elapsed)}s")
                    last_heartbeat = now

        except KeyboardInterrupt:
            print("\n  Training interrupted — saving …")

        finally:
            # --- clean shutdown -------------------------------------------
            stop_event.set()
            for p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

            # Close weight queues (non-blocking — they are Queue(maxsize=1))
            for wq in weight_queues:
                try:
                    wq.cancel_join_thread()
                    wq.close()
                except Exception:
                    pass

            # Drain exp_queue to avoid BrokenPipe warnings from background threads
            try:
                while not exp_queue.empty():
                    exp_queue.get_nowait()
            except Exception:
                pass
            try:
                exp_queue.cancel_join_thread()
                exp_queue.close()
            except Exception:
                pass

            # Drain done_queue — process any remaining DONE messages
            # so episodes_done reflects the true final count.
            try:
                while not done_queue.empty():
                    item = done_queue.get_nowait()
                    _, _wid, score, _steps = item
                    episodes_done += 1
                    scores.append(score)
                    if score > best_score:
                        best_score = score
            except Exception:
                pass
            try:
                done_queue.cancel_join_thread()
                done_queue.close()
            except Exception:
                pass

        elapsed = time.time() - start_time

        return {
            "episodes":    episodes_done,
            "best_score":  best_score,
            "avg_score":   sum(scores[-100:]) / max(len(scores[-100:]), 1),
            "total_steps": total_steps,
            "train_steps": train_steps,
            "elapsed":     elapsed,
        }
