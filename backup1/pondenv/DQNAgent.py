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
    # Assurez-vous que le masque d'action est un tensor torch
    if not isinstance(action_mask, torch.Tensor):
        action_mask = torch.tensor(action_mask, dtype=torch.bool)
    else:
        action_mask = action_mask.bool()

    if random.random() < epsilon:
        # Sélection aléatoire parmi les actions valides
        valid_indices = torch.nonzero(action_mask, as_tuple=True)[0]
        if len(valid_indices) == 0:
            raise ValueError("Aucune action valide disponible pour l'exploration.")
        action = valid_indices[torch.randint(len(valid_indices), (1,))].item()
    else:
        # Sélection gloutonne restreinte par le masque
        masked_q_values = q_values.clone()
        masked_q_values[~action_mask] = -float('inf')
        if torch.all(masked_q_values == -float('inf')):
            raise ValueError("Aucune action valide disponible pour l'exploitation.")
        action = torch.argmax(masked_q_values).item()
    return action

# Fonction d'évaluation de l'agent
def evaluate(env, q_network, num_eval_episodes=20):
    results = {'win': 0, 'lose': 0, 'tie': 0}

    for _ in range(num_eval_episodes):
        env.reset()  # Réinitialiser l'environnement
        state = torch.tensor(env.encode_state(), dtype=torch.float32)
        done = False

        while not done:
            env.generate_actions_for_current_player()
            action_mask = env.action_mask
            if not np.any(action_mask):
                break

            with torch.no_grad():
                q_values = q_network(state)
            action_idx = select_action(q_values, action_mask, epsilon=0.0)  # Toujours glouton pendant l'évaluation
            action_vector = np.zeros(env.num_actions(), dtype=int)
            action_vector[action_idx] = 1

            # Exécuter l'action
            state, reward, done = env.step_with_encoded_action(action_vector)
            state = torch.tensor(state, dtype=torch.float32)

        # Enregistrer le résultat
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

# Fonction de formation du DQN
def train_dqn(env, num_episodes=1000, gamma=0.99, alpha=0.001, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, eval_interval=10):
    state_dim = len(env.reset())  # Longueur du vecteur d'état encodé
    action_dim = env.num_actions()  # Nombre maximum d'actions possibles
    q_network = QNetwork(state_dim, action_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    epsilon = epsilon_start
    best_winrate = 0  # Pour suivre le meilleur taux de victoire
    best_model_path = 'best_q_network.pth'  # Chemin pour enregistrer le meilleur modèle

    for episode in range(num_episodes):
        env.reset()
        state = torch.tensor(env.encode_state(), dtype=torch.float32)  # État initial
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            list_of_actions = env.generate_actions_for_current_player()
            action_mask = env.action_mask
            if not np.any(action_mask):
                break

            q_values = q_network(state)

            # Sélection de l'action en utilisant la politique epsilon-greedy avec masque
            action_idx = select_action(q_values, action_mask, epsilon)
            action_vector = np.zeros(action_dim, dtype=int)
            action_vector[action_idx] = 1

            # Exécuter l'action et observer le prochain état et la récompense
            next_state, reward, done = env.step_with_encoded_action(action_vector)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

            # Calculer la valeur cible Q
            with torch.no_grad():
                target_q = reward
                if not done:
                    env.generate_actions_for_current_player()  # Mettre à jour le masque d'action pour l'état suivant
                    next_action_mask = env.action_mask
                    if np.any(next_action_mask):
                        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool)
                        next_q_values = q_network(next_state)
                        max_next_q_value = torch.max(next_q_values[next_action_mask]).item()
                    else:
                        max_next_q_value = 0
                    target_q += gamma * max_next_q_value

            # Calculer la valeur Q prédite
            predicted_q_value = q_values[action_idx]

            # Calculer la perte et effectuer la rétropropagation
            loss = loss_fn(predicted_q_value, torch.tensor(target_q, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Passer à l'état suivant
            state = next_state
            step_count += 1

        # Décroître epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Évaluation périodique
        if episode % eval_interval == 0:
            winrate, lose_rate, tie_rate = evaluate(env, q_network, num_eval_episodes=1000)
            print(f"Épisode {episode} | Taux de victoire : {winrate:.2f}% | Taux de défaite : {lose_rate:.2f}% | Taux de nul : {tie_rate:.2f}%")

            # Enregistrer le modèle si le taux de victoire s'améliore
            if winrate > best_winrate:
                best_winrate = winrate
                torch.save(q_network.state_dict(), best_model_path)
                print(f"Meilleur modèle enregistré avec un taux de victoire de {best_winrate:.2f}% à l'épisode {episode}")

    return q_network

# Fonction pour rejouer la meilleure politique
def play_with_best_policy(env, model_path='best_q_network.pth', num_episodes=10):
    state_dim = len(env.reset())
    action_dim = env.num_actions()
    q_network = QNetwork(state_dim, action_dim)
    q_network.load_state_dict(torch.load(model_path))
    q_network.eval()  # Mettre le réseau en mode évaluation

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
            action_idx = select_action(q_values, action_mask, epsilon=0.0)  # Toujours glouton

            action_vector = np.zeros(env.num_actions(), dtype=int)
            action_vector[action_idx] = 1

            state, reward, done = env.step_with_encoded_action(action_vector)
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
   # play_with_best_policy(env, model_path='best_q_network.pth', num_episodes=10000)

    # Entraîner le réseau
    trained_q_network = train_dqn(env, num_episodes=1000, eval_interval=1000)

    # Évaluation finale
    winrate, lose_rate, tie_rate = evaluate(env, trained_q_network, num_eval_episodes=1000)
    print("\n--- Résultats finaux après l'entraînement ---")
    print(f"- Taux de victoire : {winrate:.2f}%")
    print(f"- Taux de défaite : {lose_rate:.2f}%")
    print(f"- Taux de nul : {tie_rate:.2f}%")

    # Jouer avec la meilleure politique enregistrée

