# Snake Game AI 🐍🤖

Une application de comparaison d'algorithmes d'intelligence artificielle appliqués au jeu Snake classique. Ce projet permet de comparer les performances entre un algorithme de recherche de chemin déterministe (A*) et un agent d'apprentissage par renforcement (Deep Q-Network).

## 🎯 Objectifs du Projet

- **Comparer deux approches d'IA** : A* (déterministe) vs DQN (apprentissage)
- **Visualiser les performances** en temps réel avec une interface graphique moderne
- **Analyser les statistiques** de chaque agent avec des graphiques comparatifs
- **Entraîner** et sauvegarder des modèles DQN personnalisés

## 🏗️ Architecture

L'application suit une architecture modulaire basée sur le pattern MVC :

- **`game/`** : Logique du jeu (Snake, nourriture, état du jeu)
- **`ai/`** : Agents d'intelligence artificielle (A*, DQN, réseau de neurones)
- **`ui/`** : Interface utilisateur Pygame (rendu, menu, statistiques)
- **`data/`** : Persistance SQLite pour les sessions et statistiques

## 🚀 Installation et Lancement

### Prérequis

- **Python 3.10+** (recommandé : Python 3.11)
- **Git** (pour cloner le projet)

### Étape 1 : Cloner le Projet

```powershell
# Si vous n'avez pas encore cloné le projet
git clone <your-repo-url>
cd snake_game_ai
```

Ou si vous avez déjà les fichiers :

```powershell
cd c:\project\snake\snake_game_ai
```

### Étape 2 : Créer l'Environnement Virtuel

```powershell
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Alternative pour Windows Command Prompt
# .\venv\Scripts\activate.bat

# Alternative pour Git Bash sur Windows
# source venv/Scripts/activate
```

> **Note** : Si vous obtenez une erreur d'exécution de script sur PowerShell, exécutez :
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Étape 3 : Installer les Dépendances

```powershell
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer les dépendances
pip install -r requirements.txt
```

### Étape 4 : Lancer l'Application

```powershell
python main.py
```

## 🎮 Modes de Jeu

### 1. 🎮 Play (Human)
Jouez manuellement au Snake avec les touches directionnelles :
- **Flèches** ou **WASD** : Déplacer le serpent
- **P** ou **Espace** : Pause/Reprendre
- **R** : Redémarrer
- **Échap** : Quitter

### 2. 🤖 Play A*
Regardez l'agent A* jouer automatiquement :
- Algorithme de recherche de chemin optimal
- Visualisation du chemin calculé en temps réel
- Performance déterministe et prévisible

### 3. 🧠 Play DQN
Observez l'agent DQN (réseau de neurones) jouer :
- Comportement adaptatif basé sur l'apprentissage
- Affichage de l'epsilon (exploration vs exploitation)
- Chargement automatique du modèle pré-entraîné si disponible

### 4. 📚 Train DQN
Entraînez votre propre agent DQN :
- **1000 épisodes** d'entraînement par défaut
- Sauvegarde automatique du meilleur modèle
- Visualisation périodique des progrès
- **Ctrl+C** pour arrêter l'entraînement

### 5. 📊 Compare AIs / Statistics
Visualisez les statistiques comparatives :
- Graphiques des scores moyens et maximums
- Taux de victoire par agent
- Historique des performances

## 🧠 Algorithmes Implémentés

### A* (A-Star)
- **Heuristique** : Distance de Manhattan
- **Avantages** : Optimal, déterministe, rapide
- **Inconvénients** : Rigide, ne s'adapte pas

### DQN (Deep Q-Network)
- **Architecture** : Réseau feedforward (3 couches)
- **Techniques** : Experience Replay, Target Network, Double DQN
- **État** : 11 features (dangers, direction, position de la nourriture)
- **Actions** : 4 directions possibles
- **Avantages** : Adaptatif, apprend de l'expérience
- **Inconvénients** : Long à entraîner, stochastique

## 📊 Base de Données

L'application utilise SQLite pour persister :

- **GameSession** : Chaque partie jouée
- **Statistics** : Statistiques agrégées par agent
- **AgentConfig** : Configurations des modèles DQN

Les données sont stockées dans `assets/snake_game.db`.

## 🛠️ Développement

### Structure du Code

```
snake_game_ai/
├── main.py              # Point d'entrée
├── config.py            # Configuration
├── requirements.txt     # Dépendances
│
├── game/               # Logique du jeu
│   ├── point.py        # Classe Point
│   ├── direction.py    # Enum Direction
│   ├── snake.py        # Classe Snake
│   ├── food.py         # Classe Food
│   ├── game_state.py   # État du jeu
│   └── game_controller.py
│
├── ai/                 # Intelligence Artificielle
│   ├── base_agent.py   # Interface commune
│   ├── astar_agent.py  # Algorithme A*
│   ├── dqn_agent.py    # Agent DQN
│   ├── neural_network.py
│   └── replay_buffer.py
│
├── ui/                 # Interface Utilisateur
│   ├── renderer.py     # Rendu Pygame
│   ├── menu.py         # Menu principal
│   └── comparison_view.py
│
└── data/               # Persistance
    └── database_manager.py
```

### Lancer les Tests

```powershell
# Tests unitaires complets
python -m unittest discover tests

# Test spécifique
python -m unittest tests.test_snake
python -m unittest tests.test_astar
python -m unittest tests.test_dqn
```

### Configuration Avancée

Vous pouvez modifier `config.py` pour ajuster :

- **Taille de la grille** : `GRID_WIDTH`, `GRID_HEIGHT`
- **Vitesse du jeu** : `GAME_SPEED_HUMAN`, `GAME_SPEED_AI`
- **Paramètres DQN** : `DQN_CONFIG`
- **Couleurs** : `COLORS`

## 🔧 Dépannage

### Erreurs Communes

**ImportError: No module named 'pygame'**
```powershell
# Vérifiez que l'environnement virtuel est activé
.\venv\Scripts\Activate.ps1
pip install pygame
```

**CUDA out of memory (pour DQN)**
```python
# Dans config.py, forcez l'utilisation du CPU
DQN_CONFIG["force_cpu"] = True
```

**Erreur d'exécution de script PowerShell**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Performance

- **GPU** : PyTorch utilisera automatiquement CUDA si disponible
- **RAM** : L'entraînement DQN peut utiliser jusqu'à 1-2GB
- **Vitesse** : L'entraînement sur CPU est plus lent mais fonctionnel

## 📈 Résultats Attendus

Après entraînement, vous devriez observer :

- **A*** : Scores élevés et constants, comportement optimal
- **DQN** : Amélioration progressive, adaptation aux situations complexes
- **Comparaison** : A* plus efficace à court terme, DQN plus flexible à long terme

## 🤝 Contribution

Ce projet a été développé dans le cadre d'un projet académique de comparaison d'algorithmes d'IA. Les contributions sont les bienvenues !

### Améliorations Possibles

- [ ] Implémentation de Q-Learning tabulaire
- [ ] Interface web avec Flask/Django
- [ ] Modes de jeu supplémentaires (obstacles, multi-joueurs)
- [ ] Optimisations de performance
- [ ] Support de différentes tailles de grille

## 📝 Licence

Ce projet est développé à des fins éducatives dans le cadre de la Licence 3 Informatique - Université de Picardie Jules Verne.

---

**Développeurs** : HAFDANE Abderrahmane & DAHBI Meriem  
**Encadrant** : M. Yu Li  
**Année** : 2026

## 📞 Support

En cas de problème, vérifiez :

1. **Python 3.10+** est installé : `python --version`
2. **L'environnement virtuel** est activé : le prompt devrait afficher `(venv)`
3. **Les dépendances** sont installées : `pip list`
4. **Les permissions** sont correctes pour l'exécution de scripts

Pour plus d'aide, consultez la documentation des dépendances :
- [Pygame Documentation](https://www.pygame.org/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
