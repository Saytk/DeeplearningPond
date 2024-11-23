import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_utils import save_model, update_best_models


class BaseDQN:
    def __init__(self, env, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

    def select_action(self, state, action_mask):
        q_values = self.q_network(state)
        if np.random.rand() < self.epsilon:
            valid_indices = torch.nonzero(action_mask, as_tuple=True)[0]
            return valid_indices[torch.randint(len(valid_indices), (1,))].item()
        else:
            masked_q_values = q_values.clone()
            masked_q_values[~action_mask] = -float('inf')
            return torch.argmax(masked_q_values).item()

    def train_step(self, state, action, reward, next_state, done, action_mask):
        with torch.no_grad():
            target_q = reward
            if not done:
                next_q_values = self.q_network(next_state)
                max_next_q_value = torch.max(next_q_values[action_mask]).item()
                target_q += self.gamma * max_next_q_value

        predicted_q_value = self.q_network(state)[action]
        loss = self.loss_fn(predicted_q_value, torch.tensor(target_q, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def evaluate(self, num_eval_episodes=20):
        results = {'win': 0, 'lose': 0, 'tie': 0}

        for _ in range(num_eval_episodes):
            self.env.reset()
            state = torch.tensor(self.env.encode_state(), dtype=torch.float32)
            done = False

            while not done:
                self.env.generate_actions_for_current_player()
                action_mask = self.env.action_mask
                if not np.any(action_mask):
                    break

                with torch.no_grad():
                    q_values = self.q_network(state)
                action_idx = self.select_action(state, action_mask)  # Greedy during evaluation
                action_vector = np.zeros(self.action_dim, dtype=int)
                action_vector[action_idx] = 1

                state, reward, done = self.env.step_with_encoded_action(action_vector)
                state = torch.tensor(state, dtype=torch.float32)

            # Update results based on game outcome
            if self.env.board.winner == 'light':
                results['win'] += 1
            elif self.env.board.winner == 'dark':
                results['lose'] += 1
            else:
                results['tie'] += 1

        total = sum(results.values())
        winrate = results['win'] / total * 100
        lose_rate = results['lose'] / total * 100
        tie_rate = results['tie'] / total * 100

        return winrate, lose_rate, tie_rate
