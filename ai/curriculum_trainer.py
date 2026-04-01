"""Curriculum trainer — runs ParallelTrainer in stages, advancing when avg score crosses a threshold.

Stage design
────────────
Stage 1  steps_per_food=area×1  threshold=1% of cells  "Learn to find food at all"
Stage 2  steps_per_food=area×2  threshold=5% of cells  "Learn efficient navigation"
Stage 3  steps_per_food=area×4  threshold=∞             "Full game — no time pressure"

The snake plays on the same 30×25 grid throughout. Curriculum is applied via a
hard per-food step cap that relaxes between stages. This forces the agent to
learn food-finding quickly before worrying about long-range planning.

Each stage runs in chunks of CHUNK_EPISODES. After every chunk the rolling avg
(last 100 episodes) is evaluated. If it crosses the stage threshold the trainer
prints a banner and advances — no manual intervention required.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .vectorized_trainer import VectorizedTrainer


# Episodes to run per chunk before re-evaluating avg score
_CHUNK_EPISODES = 250


def _build_stages(grid_width: int, grid_height: int) -> List[Dict[str, Any]]:
    """
    Build stage definitions scaled to the grid area.

    Thresholds are expressed as fractions of total cells so the curriculum
    means the same thing regardless of grid size:
      Stage 1 threshold = 1% of cells  (e.g.  7 on 30×25)
      Stage 2 threshold = 5% of cells  (e.g. 37 on 30×25)

    steps_per_food caps are multiples of the grid area:
      Stage 1 = area×1  (tight — must eat within one grid-length of steps)
      Stage 2 = area×2  (medium)
      Stage 3 = area×4  (full game — no meaningful pressure)
    """
    area = grid_width * grid_height          # e.g. 750 for 30×25
    t1   = max(5,  area // 100)             # 1% of cells, min 5
    t2   = max(15, area //  20)             # 5% of cells, min 15
    return [
        {
            "name":           "Stage 1 — Survival & Food Finding",
            "steps_per_food": area * 1,
            "threshold":      t1,
        },
        {
            "name":           "Stage 2 — Efficient Navigation",
            "steps_per_food": area * 2,
            "threshold":      t2,
        },
        {
            "name":           "Stage 3 — Full Game",
            "steps_per_food": area * 4,
            "threshold":      99999,        # never advances — final stage
        },
    ]


class CurriculumTrainer:
    """
    Wraps ParallelTrainer with automatic stage advancement.

    Usage
    ─────
        trainer = CurriculumTrainer(agent, num_workers=4, grid_width=30, grid_height=25)
        trainer.train(num_episodes=50_000, save_path="saved_models/dqn_model.pt", ...)
    """

    def __init__(
        self,
        agent,
        n_envs:       int   = 16,
        grid_width:   int   = 30,
        grid_height:  int   = 25,
        eps_max:      float = 1.0,
        start_stage:  int   = 0,    # resume at a specific stage (0-indexed)
    ) -> None:
        self.agent       = agent
        self.n_envs      = n_envs
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.eps_max     = eps_max
        self._stages     = _build_stages(grid_width, grid_height)
        self.start_stage = max(0, min(start_stage, len(self._stages) - 1))

    # ------------------------------------------------------------------
    def train(
        self,
        num_episodes:      int  = 10_000,
        max_steps:         int  = 4500,     # absolute per-episode cap (grid×6)
        train_freq:        int  = 2,
        save_path:         str  = "saved_models/dqn_model.pt",
        checkpoint_every:  int  = 500,
        weight_sync_every: int  = 200,
        visualize_every:   int  = 0,
        visualize_fn:      Optional[Callable] = None,
        on_log:            Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run curriculum training.

        The total budget of *num_episodes* is split into chunks of
        _CHUNK_EPISODES. After each chunk the rolling avg is checked and the
        stage advances if the threshold is met.

        Args:
            num_episodes:     total episode budget across all stages.
            max_steps:        hard per-episode cap (independent of per-food cap).
            train_freq:       gradient steps per N experiences.
            save_path:        model checkpoint path.
            checkpoint_every: save every N episodes.
            weight_sync_every:broadcast weights to workers every N train steps.
            visualize_every:  render one episode every N episodes (0 = off).
            visualize_fn:     callback(agent) for visual episodes.
            on_log:           callback(ep, total, score, avg, best, stats, elapsed, eta).

        Returns:
            Dict with keys: episodes, best_score, avg_score, stage,
            total_steps, train_steps, elapsed.
        """
        stage_idx      = self.start_stage
        episodes_done  = 0
        best_score     = 0
        all_scores:    List[int] = []
        total_steps    = 0
        train_steps    = 0
        global_start   = time.time()

        def _fmt(secs: float) -> str:
            secs = int(secs)
            h, r = divmod(secs, 3600)
            m, s = divmod(r, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        self._print_stage_banner(self._stages, stage_idx, episodes_done)
        aborted = False

        try:
            while episodes_done < num_episodes:
                stage          = self._stages[stage_idx]
                steps_per_food = stage["steps_per_food"]
                chunk          = min(_CHUNK_EPISODES, num_episodes - episodes_done)

                # ── build a VectorizedTrainer for this chunk ──────────────────
                trainer = VectorizedTrainer(
                    agent       = self.agent,
                    n_envs      = self.n_envs,
                    grid_width  = self.grid_width,
                    grid_height = self.grid_height,
                    eps_max     = self.eps_max,
                )

                # Capture the current global state for this chunk's log callback.
                # Use a local variable (not a closure over `episodes_done`) to
                # avoid the classic late-binding bug that caused duplicate logs.
                ep_offset   = episodes_done
                global_best = best_score

                def _make_chunk_log(offset, g_best):
                    def _chunk_log(ep, total, score, avg_chunk, best_ep,
                                   stats, elapsed, eta):
                        # Use the global rolling avg (all_scores populated via
                        # on_episode every episode, not just at log boundaries)
                        global_avg = (sum(all_scores[-100:])
                                      / max(len(all_scores[-100:]), 1))
                        if on_log:
                            on_log(
                                offset + ep, num_episodes,
                                score, global_avg,
                                max(g_best, best_ep),
                                stats, elapsed, eta,
                            )
                    return _chunk_log

                def _on_episode(score):
                    all_scores.append(score)

                results = trainer.train(
                    num_episodes      = chunk,
                    max_steps         = max_steps,
                    steps_per_food    = steps_per_food,
                    train_freq        = train_freq,
                    save_path         = save_path,
                    checkpoint_every  = checkpoint_every,
                    weight_sync_every = weight_sync_every,
                    visualize_every   = visualize_every,
                    visualize_fn      = visualize_fn,
                    on_log            = _make_chunk_log(ep_offset, global_best),
                    on_episode        = _on_episode,
                )

                episodes_done += results["episodes"]
                total_steps   += results["total_steps"]
                train_steps   += results["train_steps"]
                if results["best_score"] > best_score:
                    best_score = results["best_score"]

                if results.get("aborted"):
                    aborted = True
                    elapsed = time.time() - global_start
                    print(f"\n  [Curriculum] Interrupted at ep {episodes_done} — "
                          f"elapsed: {_fmt(elapsed)}")
                    break

                # ── evaluate rolling avg and decide whether to advance ────────
                recent  = all_scores[-100:]
                avg     = sum(recent) / max(len(recent), 1)
                elapsed = time.time() - global_start

                print(
                    f"  [Curriculum] ep {episodes_done}/{num_episodes} | "
                    f"Stage {stage_idx + 1} | "
                    f"avg(100): {avg:.1f} / {stage['threshold']} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"elapsed: {_fmt(elapsed)}"
                )

                if avg >= stage["threshold"] and stage_idx < len(self._stages) - 1:
                    stage_idx += 1
                    self._print_stage_banner(self._stages, stage_idx, episodes_done)

        except KeyboardInterrupt:
            # Fallback: interrupt outside VectorizedTrainer (e.g. during setup)
            aborted = True
            elapsed = time.time() - global_start
            print(f"\n  [Curriculum] Interrupted at ep {episodes_done} — "
                  f"elapsed: {_fmt(elapsed)}")

        return {
            "episodes":    episodes_done,
            "best_score":  best_score,
            "avg_score":   sum(all_scores[-100:]) / max(len(all_scores[-100:]), 1),
            "stage":       stage_idx + 1,
            "total_steps": total_steps,
            "train_steps": train_steps,
            "elapsed":     time.time() - global_start,
            "aborted":     aborted,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _print_stage_banner(stages: list, stage_idx: int, episodes_done: int) -> None:
        stage = stages[stage_idx]
        W = 60
        print(f"\n{'='*W}")
        print(f"  CURRICULUM  →  {stage['name']}")
        print(f"  steps/food cap : {stage['steps_per_food']}")
        print(f"  advance when   : avg(100) > {stage['threshold']}")
        print(f"  episodes so far: {episodes_done}")
        print(f"{'='*W}\n")
