import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from PondEnv import PondEnv

# Définition du réseau Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Fonction de sélection d'action avec masque
def select_action(q_values, action_mask, epsilon):
    """
    Sélectionne une action en utilisant une stratégie epsilon-greedy avec un masque d'action.

    Args:
        q_values (torch.Tensor): Tensor des valeurs Q pour chaque action.
        action_mask (np.ndarray ou torch.Tensor): Masque binaire indiquant les actions valides (1) et invalides (0).
        epsilon (float): Probabilité d'exploration (sélection d'une action aléatoire).

    Returns:
        int: Index de l'action sélectionnée.
    """
    if not isinstance(action_mask, torch.Tensor):
        action_mask = torch.tensor(action_mask, dtype=torch.bool)
    else:
        action_mask = action_mask.bool()

    if random.random() < epsilon:
        valid_indices = torch.nonzero(action_mask, as_tuple=True)[0]
        if len(valid_indices) == 0:
            raise ValueError("Aucune action valide disponible pour l'exploration.")
        action = valid_indices[torch.randint(len(valid_indices), (1,))].item()
    else:
        masked_q_values = q_values.clone()
        masked_q_values[~action_mask] = -float('inf')
        if torch.all(masked_q_values == -float('inf')):
            raise ValueError("Aucune action valide disponible pour l'exploitation.")
        action = torch.argmax(masked_q_values).item()
    return action

# Fonction d'évaluation de l'agent
def evaluate(env, q_network, num_eval_episodes=20):
    results = {'win': 0, 'lose': 0, 'tie': 0}

    for i in range(num_eval_episodes):
        print("Épisode d'évaluation :", i + 1)
        env.reset()
        state = torch.tensor(env.encode_state(), dtype=torch.float32)
        done = False

        while not done:
            env.generate_actions_for_current_player()
            action_mask = env.action_mask
            if not np.any(action_mask):
                break

            with torch.no_grad():
                q_values = q_network(state)
            action_idx = select_action(q_values, action_mask, epsilon=0.0)
            action = env.decode_action(action_idx)

            state, reward, done = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)

        if env.board.winner == 'light':
            results['win'] += 1
        elif env.board.winner == 'dark':
            results['lose'] += 1
        else:
            results['tie'] += 1

    total = sum(results.values())
    winrate = results['win'] / total * 100
    lose_rate = results['lose'] / total * 100
    tie_rate = results['tie'] / total * 100

    return winrate, lose_rate, tie_rate

# Fonction de formation du DQN avec gestion des 5 meilleurs modèles
def train_dqn(env, num_episodes=1000, gamma=0.99, alpha=0.001, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, eval_interval=10):
    state_dim = len(env.reset())
    action_dim = env.num_actions()
    q_network = QNetwork(state_dim, action_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    epsilon = epsilon_start
    best_models = []  # Liste des meilleurs modèles [(winrate, model_path)]

    for episode in range(num_episodes):
        env.reset()
        state = torch.tensor(env.encode_state(), dtype=torch.float32)
        total_reward = 0
        done = False

        while not done:
            action_mask = env.action_mask
            if not np.any(action_mask):
                break

            q_values = q_network(state)
            action_idx = select_action(q_values, action_mask, epsilon)
            action = env.decode_action(action_idx)

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

            with torch.no_grad():
                target_q = reward
                if not done:
                    next_action_mask = env.action_mask
                    if np.any(next_action_mask):
                        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool)
                        next_q_values = q_network(next_state)
                        max_next_q_value = torch.max(next_q_values[next_action_mask]).item()
                    else:
                        max_next_q_value = 0
                    target_q += gamma * max_next_q_value

            predicted_q_value = q_values[action_idx]
            loss = loss_fn(predicted_q_value, torch.tensor(target_q, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Évaluer périodiquement le modèle
        if episode % eval_interval == 0:
            winrate, lose_rate, tie_rate = evaluate(env, q_network, num_eval_episodes=100)
            print(f"Épisode {episode} | Taux de victoire : {winrate:.2f}% | Taux de défaite : {lose_rate:.2f}% | Taux de nul : {tie_rate:.2f}%")

            # Ajouter le modèle si son winrate est suffisamment élevé
            model_path = f"q_network_{winrate:.2f}.pth"
            if len(best_models) < 5 or winrate > min(best_models, key=lambda x: x[0])[0]:
                # Sauvegarder le modèle
                torch.save(q_network.state_dict(), model_path)
                print(f"Modèle enregistré : {model_path}")

                # Ajouter le modèle à la liste des meilleurs
                best_models.append((winrate, model_path))
                best_models = sorted(best_models, key=lambda x: x[0], reverse=True)[:5]

                # Supprimer les anciens modèles s'ils ne font plus partie du top 5
                for _, path in best_models[5:]:
                    if os.path.exists(path):
                        os.remove(path)

    print("\n--- Top 5 des modèles ---")
    for winrate, path in best_models:
        print(f"Modèle : {path} | Winrate : {winrate:.2f}%")

    return q_network

# Fonction pour rejouer avec la meilleure politique
def play_with_best_policy(env, model_path='best_q_network.pth', num_episodes=10):
    state_dim = len(env.reset())
    action_dim = env.num_actions()
    q_network = QNetwork(state_dim, action_dim)
    q_network.load_state_dict(torch.load(model_path))
    q_network.eval()

    total_wins = 0
    total_losses = 0
    total_ties = 0

    for _ in range(num_episodes):
        env.reset()
        state = torch.tensor(env.encode_state(), dtype=torch.float32)
        done = False

        while not done:
            env.generate_actions_for_current_player()
            action_mask = env.action_mask
            if not np.any(action_mask):
                break

            with torch.no_grad():
                q_values = q_network(state)
            action_idx = select_action(q_values, action_mask, epsilon=0.0)
            action = env.decode_action(action_idx)

            state, reward, done = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)

        if env.board.winner == 'light':
            total_wins += 1
        elif env.board.winner == 'dark':
            total_losses += 1
        else:
            total_ties += 1

    total_games = total_wins + total_losses + total_ties
    print(f"\nRésultats après avoir joué avec la meilleure politique sur {total_games} parties :")
    print(f"- Victoires : {total_wins}")
    print(f"- Défaites : {total_losses}")
    print(f"- Nuls : {total_ties}")

# Main
if __name__ == "__main__":
    env = PondEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Entraîner le réseau
    trained_q_network = train_dqn(env, num_episodes=1000, eval_interval=100)

    # Évaluation finale
    winrate, lose_rate, tie_rate = evaluate(env, trained_q_network, num_eval_episodes=100)
    print("\n--- Résultats finaux après l'entraînement ---")
    print(f"- Taux de victoire : {winrate:.2f}%")
    print(f"- Taux de défaite : {lose_rate:.2f}%")
    print(f"- Taux de nul : {tie_rate:.2f}%")

    # Jouer avec la meilleure politique enregistrée
    play_with_best_policy(env, model_path='best_q_network.pth', num_episodes=10)
