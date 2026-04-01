"""
Snake Game AI - Point d'entrée principal
=========================================

Application de comparaison d'algorithmes d'IA pour le jeu Snake:
- A* (recherche de chemin déterministe)
- DQN (Deep Q-Network - apprentissage par renforcement)

Usage:
    python main.py
"""

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import sys
import pygame
import torch

from config import (
    GRID_WIDTH, GRID_HEIGHT, GRID_SIZE, HUD_HEIGHT,
    GAME_SPEED_HUMAN, GAME_SPEED_AI, GAME_SPEED_TRAINING,
    DQN_CONFIG, TRAINING_CONFIG, DATABASE_PATH, CURRICULUM_STAGES,
)
from game import GameState, GameController, Direction
from game.game_controller import GameMode
from ai import AStarAgent, DQNAgent
from ui import Renderer, Menu, ComparisonView, TrainingConfigView
from ui.menu import MenuOption
from data import DatabaseManager


def play_human_mode(db_manager: DatabaseManager) -> None:
    """Lance le jeu en mode humain."""
    game_state = GameState(GRID_WIDTH, GRID_HEIGHT)
    renderer = Renderer(GRID_WIDTH, GRID_HEIGHT, GRID_SIZE)
    
    controller = GameController(
        game_state=game_state,
        renderer=renderer,
        agent=None,
        db_manager=db_manager,
        mode=GameMode.HUMAN,
        game_speed=GAME_SPEED_HUMAN
    )
    
    controller.run()


def play_astar_mode(db_manager: DatabaseManager) -> None:
    """Lance le jeu avec l'agent A*."""
    game_state = GameState(GRID_WIDTH, GRID_HEIGHT)
    renderer = Renderer(GRID_WIDTH, GRID_HEIGHT, GRID_SIZE)
    agent = AStarAgent()
    
    controller = GameController(
        game_state=game_state,
        renderer=renderer,
        agent=agent,
        db_manager=db_manager,
        mode=GameMode.ASTAR,
        game_speed=GAME_SPEED_AI
    )
    
    controller.run()


def play_dqn_mode(db_manager: DatabaseManager) -> None:
    """Lance le jeu avec l'agent DQN pré-entraîné."""
    game_state = GameState(GRID_WIDTH, GRID_HEIGHT)
    renderer = Renderer(GRID_WIDTH, GRID_HEIGHT, GRID_SIZE)
    
    agent = DQNAgent(**DQN_CONFIG)
    
    # Essayer de charger un modèle existant
    try:
        agent.load_model("saved_models/dqn_model.pt")
        agent.set_eval_mode()
        print("Loaded pre-trained DQN model")
    except FileNotFoundError:
        print("No pre-trained model found. Using random policy.")
        agent.epsilon = 0.1
    except RuntimeError as e:
        print(f"Incompatible checkpoint (architecture changed): {e}")
        print("No model loaded — train first with the new architecture.")
        agent.set_eval_mode()
    
    controller = GameController(
        game_state=game_state,
        renderer=renderer,
        agent=agent,
        db_manager=db_manager,
        mode=GameMode.DQN,
        game_speed=GAME_SPEED_AI
    )
    
    controller.run()


def train_dqn_mode(db_manager: DatabaseManager) -> None:
    """Entraîne l'agent DQN."""
    import os
    import time as _time

    # --- Show config screen -------------------------------------------------
    screen_width  = GRID_WIDTH  * GRID_SIZE
    screen_height = GRID_HEIGHT * GRID_SIZE + HUD_HEIGHT
    config_view   = TrainingConfigView(screen_width, screen_height)
    user_cfg      = config_view.show()

    if user_cfg is None:
        return  # User pressed BACK / ESC

    num_episodes    = user_cfg.get("num_episodes",          TRAINING_CONFIG["num_episodes"])
    max_steps       = user_cfg.get("max_steps_per_episode", TRAINING_CONFIG["max_steps_per_episode"])
    visualize_every = user_cfg.get("visualize_every",       TRAINING_CONFIG["visualize_every"])
    decay_type      = user_cfg.get("epsilon_decay_type",    DQN_CONFIG["epsilon_decay_type"])
    train_freq      = user_cfg.get("train_freq",             TRAINING_CONFIG.get("train_freq", 4))
    device_choice   = user_cfg.get("device", None)
    num_workers     = user_cfg.get("num_workers", 1)

    # Merge user choices into agent config
    agent_cfg = {**DQN_CONFIG, "epsilon_decay_type": decay_type, "device": device_choice}

    # --- Setup --------------------------------------------------------------
    game_state = GameState(GRID_WIDTH, GRID_HEIGHT)
    renderer   = Renderer(GRID_WIDTH, GRID_HEIGHT, GRID_SIZE)
    agent      = DQNAgent(**agent_cfg)

    model_path  = "saved_models/dqn_model.pt"
    buffer_path = "saved_models/dqn_buffer.npz"
    resuming    = False
    try:
        agent.load_model(model_path)
        print("Resuming training from existing model")
        resuming = True
    except FileNotFoundError:
        print("Starting training from scratch")
    except RuntimeError as e:
        print(f"Incompatible checkpoint (architecture changed): {e}")
        print("Starting training from scratch with fresh weights")

    if resuming:
        loaded = agent.load_buffer(buffer_path)
        if not loaded:
            print("  [Buffer] No compatible buffer found — starting empty")

    eps_max = 0.5 if resuming else 1.0

    device_label = str(agent.device).upper().replace(":", " ").replace("CUDA 0", "CUDA — " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))
    W = 60
    print(f"\n{'='*W}")
    print(f"  DQN Training  —  Device: {device_label}")
    print(f"{'='*W}")
    print(f"  Episodes    : {num_episodes:,}")
    print(f"  Max steps   : {max_steps:,}  |  Decay: {decay_type}")
    print(f"  Train freq  : every {train_freq} step{'s' if train_freq > 1 else ''}")
    print(f"  Visualize   : {'OFF' if visualize_every == 0 else f'every {visualize_every} ep'}")
    if num_workers > 1:
        print(f"  Envs        : {num_workers} vectorized (single process, batch GPU)")
    else:
        print(f"  Envs        : serial")
    print(f"{'='*W}\n")

    # Timing helper
    def _fmt_time(secs: float) -> str:
        secs = int(secs)
        h, rem = divmod(secs, 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    def _log_line(ep, total, score, avg, best, stats, elapsed, eta):
        eps_str = f"ε: {stats['epsilon']:.3f} | " if num_workers <= 1 else ""
        print(f"Ep {ep:4d}/{total} | "
              f"Score: {score:3d} | Avg: {avg:5.1f} | Best: {best:3d} | "
              f"{eps_str}lr: {stats['lr']:.2e} | "
              f"Elapsed: {_fmt_time(elapsed)} | ETA: {_fmt_time(eta)}")

    aborted      = False
    best_score   = 0
    episodes_run = 0
    total_time   = 0.0

    try:
        # ==================================================================
        # Vectorized training  (num_workers > 1)  — curriculum mode
        # Uses n_envs game instances in a single process with batch GPU
        # inference.  No IPC, no pickle, no queue — GPU ~70-85% busy.
        # ==================================================================
        if num_workers > 1:
            from ai.curriculum_trainer import CurriculumTrainer

            n_envs = num_workers   # UI "ENVS" slider maps directly to n_envs

            trainer = CurriculumTrainer(
                agent       = agent,
                n_envs      = n_envs,
                grid_width  = GRID_WIDTH,
                grid_height = GRID_HEIGHT,
                eps_max     = eps_max,
                start_stage = 0 if not resuming else 2,
            )

            def _visualize(a):
                a.set_eval_mode()
                game_state.reset()
                vis_ok = True
                while vis_ok and not game_state.game_over:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            vis_ok = False
                        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            vis_ok = False
                    action = a.get_action(game_state)
                    game_state.update(action)
                    renderer.render(game_state, a)
                    renderer.clock.tick(30)
                a.set_train_mode()

            print(f"  Envs        : {n_envs} vectorized  |  batch={agent.batch_size}")

            results = trainer.train(
                num_episodes    = num_episodes,
                max_steps       = max_steps,
                train_freq      = train_freq,
                save_path       = model_path,
                visualize_every = visualize_every,
                visualize_fn    = _visualize if visualize_every > 0 else None,
                on_log          = _log_line,
            )

            best_score   = results["best_score"]
            total_time   = results["elapsed"]
            episodes_run = results["episodes"]
            aborted      = results.get("aborted", episodes_run < num_episodes)

        # ==================================================================
        # Serial training  (num_workers == 1)
        # ==================================================================
        else:
            controller = GameController(
                game_state=game_state, renderer=renderer, agent=agent,
                db_manager=db_manager, mode=GameMode.DQN_TRAINING, game_speed=0,
            )
            scores: list = []
            train_start = _time.time()
            ep_times: list = []

            for episode in range(num_episodes):
                ep_t0 = _time.time()
                score, _ = controller.run_training_episode(
                    max_steps=max_steps, train_freq=train_freq)
                ep_times.append(_time.time() - ep_t0)
                if len(ep_times) > 50:
                    ep_times.pop(0)
                scores.append(score)

                if score > best_score:
                    best_score = score
                    os.makedirs("saved_models", exist_ok=True)
                    agent.save_model(model_path)

                if (episode + 1) % 10 == 0:
                    avg_score = sum(scores[-100:]) / min(len(scores), 100)
                    stats     = agent.get_stats()
                    elapsed   = _time.time() - train_start
                    avg_ep_t  = sum(ep_times) / len(ep_times)
                    remaining = avg_ep_t * (num_episodes - episode - 1)
                    _log_line(episode + 1, num_episodes, score,
                              avg_score, best_score, stats, elapsed, remaining)
                    agent.update_scheduler(avg_score)

                if (episode + 1) % 100 == 0:
                    agent.save_model(model_path)
                    print(f"  ↳ Checkpoint saved at episode {episode + 1}")

                if visualize_every > 0 and (episode + 1) % visualize_every == 0:
                    game_state.reset()
                    agent.set_eval_mode()
                    running = True
                    while running and not game_state.game_over:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                aborted = True
                            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                                running = False
                                aborted = True
                        action = agent.get_action(game_state)
                        game_state.update(action)
                        renderer.render(game_state, agent)
                        renderer.clock.tick(30)
                    agent.set_train_mode()
                    if aborted:
                        break

            total_time   = _time.time() - train_start
            episodes_run = len(scores)

    except KeyboardInterrupt:
        print("\n  Training interrupted — saving...")
        aborted = True

    # --- Final summary (both paths) — always reached via finally semantics ---
    os.makedirs("saved_models", exist_ok=True)
    agent.save_model(model_path)
    agent.save_buffer(buffer_path)
    print(f"\n{'='*W}")
    print(f"  {'Interrupted' if aborted else 'Training Complete'}!")
    print(f"  Episodes run : {episodes_run:,} / {num_episodes:,}")
    print(f"  Best Score   : {best_score}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    print(f"  Total time   : {_fmt_time(total_time)}")
    print(f"  Model saved  : {model_path}")
    print(f"{'='*W}")


def play_versus_mode(db_manager: DatabaseManager) -> None:
    """Lance le mode A* vs DQN en écran partagé."""
    from ui.split_screen_view import SplitScreenView
    view = SplitScreenView(db_manager)
    view.run()


def compare_mode(db_manager: DatabaseManager) -> None:
    """Lance la comparaison côte-à-côte des IAs."""
    # Pour cette démo, on montre les statistiques
    show_statistics(db_manager)


def show_statistics(db_manager: DatabaseManager) -> None:
    """Affiche les statistiques comparatives."""
    # Récupérer les statistiques
    stats = db_manager.get_statistics()
    
    # Ajouter les types d'agents manquants
    for agent_type in ['human', 'astar', 'dqn']:
        if agent_type not in stats:
            stats[agent_type] = None
    
    # Afficher la vue de comparaison
    screen_width = GRID_WIDTH * GRID_SIZE
    screen_height = GRID_HEIGHT * GRID_SIZE + HUD_HEIGHT
    
    comparison_view = ComparisonView(screen_width, screen_height)
    comparison_view.show(stats)


def main() -> None:
    """Point d'entrée principal."""
    # Initialiser Pygame
    pygame.init()
    
    # Créer le gestionnaire de base de données
    db_manager = DatabaseManager(DATABASE_PATH)
    
    # Dimensions de la fenêtre pour le menu
    screen_width = GRID_WIDTH * GRID_SIZE
    screen_height = GRID_HEIGHT * GRID_SIZE + HUD_HEIGHT
    
    running = True
    
    while running:
        # Afficher le menu
        menu = Menu(screen_width, screen_height)
        option = menu.run()
        
        # Traiter le choix
        if option == MenuOption.PLAY_HUMAN:
            play_human_mode(db_manager)
        
        elif option == MenuOption.PLAY_ASTAR:
            play_astar_mode(db_manager)
        
        elif option == MenuOption.PLAY_DQN:
            play_dqn_mode(db_manager)

        elif option == MenuOption.PLAY_VERSUS:
            play_versus_mode(db_manager)

        elif option == MenuOption.TRAIN_DQN:
            train_dqn_mode(db_manager)
        
        elif option == MenuOption.COMPARE:
            compare_mode(db_manager)
        
        elif option == MenuOption.STATISTICS:
            show_statistics(db_manager)
        
        elif option == MenuOption.QUIT:
            running = False
    
    # Nettoyage
    db_manager.disconnect()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
