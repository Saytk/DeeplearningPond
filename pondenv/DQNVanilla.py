import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from PondEnv import PondEnv  # Assuming PondEnv is defined as provided
import matplotlib.pyplot as plt


# Define the neural network
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x.float()  # Ensure the input is of type float
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Main DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        self.batch_size = 64
        self.update_target_every = 10
        self.step_counter = 0

    def select_action(self, state, mask):
        """
        Sélectionne une action en utilisant epsilon-greedy et un masque d'actions valides.

        :param state: L'état encodé (vecteur d'entrée pour le réseau).
        :param mask: Masque des actions valides (1 pour valide, 0 pour invalide).
        :return: Index de l'action choisie.
        """
        if np.random.rand() <= self.epsilon:
            # Choisir une action valide au hasard
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) > 0:
                return random.choice(valid_actions)
            else:
                raise ValueError("Aucune action valide disponible dans le masque.")
        else:
            # Prédire les valeurs Q des actions avec le réseau de politique
            state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().squeeze()

            # Appliquer le masque aux Q-values
            masked_q_values = q_values * mask

            # Renormaliser les probabilités
            total = masked_q_values.sum()
            if total > 0:
                probabilities = masked_q_values / total
            else:
                raise ValueError("Aucune action valide disponible après masquage.")

            # Choisir une action basée sur les probabilités
            action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
            return action_idx

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.push(state, action_idx, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample(self.batch_size)
        batch_state = torch.tensor([s for s, a, r, ns, d in minibatch], device=self.device)
        batch_action = torch.tensor([a for s, a, r, ns, d in minibatch], device=self.device)
        batch_reward = torch.tensor([r for s, a, r, ns, d in minibatch], device=self.device)
        batch_next_state = torch.tensor([ns for s, a, r, ns, d in minibatch], device=self.device)
        batch_done = torch.tensor([d for s, a, r, ns, d in minibatch], device=self.device)

        current_q_values = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(batch_next_state).max(1)[0]
        expected_q_values = batch_reward + (1 - batch_done.float()) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Updating PondEnv to include action indexing
class PondEnvWithActionIndex(PondEnv):
    def __init__(self):
        super().__init__()
        self.build_action_space()

    def build_action_space(self):
        self.action_space = []
        grid_size = self.board.grid_size
        # Generate all possible place actions
        for row in range(grid_size):
            for col in range(grid_size):
                action = {
                    "type": "place",
                    "row": row,
                    "col": col,
                    "piece": "Egg"
                }
                self.action_space.append(action)
        # Generate all possible move actions
        for start_row in range(grid_size):
            for start_col in range(grid_size):
                for end_row in range(grid_size):
                    for end_col in range(grid_size):
                        action = {
                            "type": "move",
                            "start_row": start_row,
                            "start_col": start_col,
                            "end_row": end_row,
                            "end_col": end_col
                        }
                        self.action_space.append(action)
        # Create mapping from action to index and vice versa
        self.action_to_idx = {self.action_to_str(a): idx for idx, a in enumerate(self.action_space)}
        self.idx_to_action = {idx: a for idx, a in enumerate(self.action_space)}

    def action_to_str(self, action):
        return str(action)

    def action_to_index(self, action):
        return self.action_to_idx.get(self.action_to_str(action), None)

    def index_to_action(self, idx):
        return self.idx_to_action.get(idx, None)

    def encode_action(self, action):
        return PondEnv.encode_action(action, self.board.grid_size)

    def encode_state(self):
        return super().encode_state()

    def num_actions(self):
        return len(self.action_space)


# Training the DQN Agent
def train_dqn(episodes=500):
    env = PondEnvWithActionIndex()
    state_size = len(env.encode_state())
    action_size = env.num_actions()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(state_size, action_size, device)
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            available_actions = env.available_actions()
            if not available_actions:
                break

            state_encoded = env.encode_state()
            mask = env.action_mask  # Use the action mask from the environment
            action_idx = agent.select_action(state_encoded, mask)

            # Take a step in the environment
            action = env.index_to_action(action_idx)
            next_state, reward, done = env.step(action, play_random_after_agent=True)
            next_state_encoded = env.encode_state()

            # Remember the experience
            agent.remember(state_encoded, action_idx, reward, next_state_encoded, done)

            # Replay experience
            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                agent.update_target_network()
                agent.update_epsilon()
                break

        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Plotting the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Agent Performance')
    plt.show()


if __name__ == "__main__":
    train_dqn()
