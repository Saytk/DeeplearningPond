import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from PondEnv import PondEnv


# Define the Q-Network
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


# Masked action selection
def select_action(q_values, action_mask, epsilon):
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


# Evaluation function
def evaluate(env, q_network, num_eval_episodes=20):
    results = {'win': 0, 'lose': 0, 'tie': 0}

    for _ in range(num_eval_episodes):
        env = PondEnv()
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
            action_vector = np.zeros(env.num_actions(), dtype=int)
            action_vector[action_idx] = 1

            state, reward, done = env.step_with_encoded_action(action_vector)
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


# DQN Training Loop with Encoded Action and Top 5 Models
def train_dqn(env, num_episodes=1000, gamma=0.99, alpha=0.001, epsilon=1, epsilon_min=0.1, epsilon_decay=0.995, eval_interval=10):
    state_dim = len(env.reset())
    action_dim = env.num_actions()
    q_network = QNetwork(state_dim, action_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    epsilon = epsilon
    best_models = []  # Liste des 5 meilleurs modèles [(winrate, model_path)]

    for episode in range(num_episodes):
        env = PondEnv()
        state = torch.tensor(env.encode_state(), dtype=torch.float32)
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            list_of_actions = env.generate_actions_for_current_player()
            action_mask = env.action_mask
            if not np.any(action_mask):
                break

            q_values = q_network(state)

            action_idx = select_action(q_values, action_mask, epsilon)
            action_vector = np.zeros(action_dim, dtype=int)
            action_vector[action_idx] = 1

            next_state, reward, done = env.step_with_encoded_action(action_vector)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

            with torch.no_grad():
                target_q = reward
                if not done:
                    next_q_values = q_network(next_state)
                    target_q += gamma * torch.max(next_q_values[torch.tensor(action_mask, dtype=torch.bool)]).item()

            q_values = q_network(state)
            predicted_q_value = q_values[action_idx]

            loss = loss_fn(predicted_q_value, torch.tensor(target_q, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            step_count += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Évaluation périodique
        if episode % eval_interval == 0:
            winrate, lose_rate, tie_rate = evaluate(env, q_network, num_eval_episodes=1000)
            print(f"Episode {episode} | Win Rate: {winrate:.2f}% | Lose Rate: {lose_rate:.2f}% | Tie Rate: {tie_rate:.2f}%")

            model_path = f"q_network_{winrate:.2f}.pth"
            if len(best_models) < 5 or winrate > min(best_models, key=lambda x: x[0])[0]:
                # Sauvegarder le modèle
                torch.save(q_network.state_dict(), model_path)
                print(f"Modèle enregistré : {model_path}")

                # Ajouter au top 5
                best_models.append((winrate, model_path))
                best_models = sorted(best_models, key=lambda x: x[0], reverse=True)[:5]

                # Supprimer les anciens fichiers
                for _, path in best_models[5:]:
                    if os.path.exists(path):
                        os.remove(path)

    print("\n--- Top 5 des modèles ---")
    for winrate, path in best_models:
        print(f"Modèle : {path} | Winrate : {winrate:.2f}%")

    return q_network


# Main
if __name__ == "__main__":
    env = PondEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the network
    trained_q_network = train_dqn(env, num_episodes=10000, eval_interval=10)

    # Final evaluation
    env = PondEnv()
    winrate, lose_rate, tie_rate = evaluate(env, trained_q_network, num_eval_episodes=1000)
    print("\n--- Final Results After Training ---")
    print(f"- Win Rate: {winrate:.2f}%")
    print(f"- Lose Rate: {lose_rate:.2f}%")
    print(f"- Tie Rate: {tie_rate:.2f}%")