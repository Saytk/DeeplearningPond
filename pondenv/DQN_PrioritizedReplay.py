import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from PondEnv import PondEnv
from collections import deque


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


# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)

        weights = (total * probabilities[indices]) ** -beta
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# Masked action selection
def select_action(q_values, action_mask, epsilon):
    action_mask = torch.tensor(action_mask, dtype=torch.bool) if not isinstance(action_mask, torch.Tensor) else action_mask.bool()

    if random.random() < epsilon:
        valid_indices = torch.nonzero(action_mask, as_tuple=True)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid actions for exploration.")
        action = valid_indices[torch.randint(len(valid_indices), (1,))].item()
    else:
        masked_q_values = q_values.clone()
        masked_q_values[~action_mask] = -float('inf')
        if torch.all(masked_q_values == -float('inf')):
            raise ValueError("No valid actions for exploitation.")
        action = torch.argmax(masked_q_values).item()
    return action


# Evaluation function
def evaluate(env, q_network, num_eval_episodes=200):
    results = {'win': 0, 'lose': 0, 'tie': 0}

    for _ in range(num_eval_episodes):
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


# DQN Training Loop with PER
def train_dqn_with_per(env, num_episodes=1000, gamma=0.99, alpha=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, target_update_freq=100, batch_size=64, eval_interval=10, buffer_size=10000, beta_start=0.4, beta_increment=1e-4):
    state_dim = len(env.reset())
    action_dim = env.num_actions()
    q_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    replay_buffer = PrioritizedReplayBuffer(buffer_size)

    beta = beta_start
    best_models = []

    for episode in range(num_episodes):
        env.reset()
        state = torch.tensor(env.encode_state(), dtype=torch.float32)
        done = False

        while not done:
            env.generate_actions_for_current_player()
            action_mask = env.action_mask
            if not np.any(action_mask):
                break

            q_values = q_network(state)
            action_idx = select_action(q_values, action_mask, epsilon)
            action_vector = np.zeros(action_dim, dtype=int)
            action_vector[action_idx] = 1

            next_state, reward, done = env.step_with_encoded_action(action_vector)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            replay_buffer.store(state.numpy(), action_idx, reward, next_state.numpy(), done)
            state = next_state

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size, beta)

                with torch.no_grad():
                    next_q_values = target_network(next_states)
                    max_next_q_values = next_q_values.max(1)[0]
                    targets = rewards + gamma * (1 - dones) * max_next_q_values

                q_values = q_network(states)
                q_current = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                td_errors = torch.abs(q_current - targets).detach().numpy()
                replay_buffer.update_priorities(indices, td_errors)

                loss = (weights * (q_current - targets) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        beta = min(1.0, beta + beta_increment)

        if episode % eval_interval == 0:
            winrate, lose_rate, tie_rate = evaluate(env, q_network, num_eval_episodes=200)
            print(f"Episode {episode} | Win Rate: {winrate:.2f}% | Lose Rate: {lose_rate:.2f}% | Tie Rate: {tie_rate:.2f}%")

    return q_network


if __name__ == "__main__":
    env = PondEnv()
    trained_q_network = train_dqn_with_per(env, num_episodes=10000, eval_interval=20)
