import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from PondEnv import PondEnv  # Assurez-vous que PondEnv est correctement importé

# Définition du réseau neuronal pour approximer la fonction Q
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Agent DQN
class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state, action_mask):
        if np.random.rand() <= self.epsilon:
            # Sélectionne une action valide aléatoire
            env.generate_actions_for_current_player()
            action_mask = env.action_mask
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state)
                q_values = q_values.cpu().numpy().flatten()
                # Applique le masque d'actions pour sélectionner uniquement les actions valides
                masked_q_values = np.ma.array(q_values, mask=1 - action_mask)
                return np.argmax(masked_q_values)

    def optimize_model(self, state, action, reward, next_state, done, action_mask):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # Calcul de la valeur Q cible
        self.model.eval()
        with torch.no_grad():
            next_q_values = self.model(next_state)
            next_q_values = next_q_values.cpu().numpy().flatten()
            # Applique le masque d'actions pour les prochaines actions valides
            masked_next_q_values = np.ma.array(next_q_values, mask=1 - action_mask)
            next_q_value = np.max(masked_next_q_values)
            next_q_value = torch.FloatTensor([next_q_value]).to(self.device)

        target_q_value = reward + (1 - done) * self.gamma * next_q_value

        # Calcul de la valeur Q prédite
        self.model.train()
        q_values = self.model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Mise à jour du modèle
        loss = self.criterion(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Décroissance epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Boucle d'entraînement
if __name__ == "__main__":
    num_episodes = 500
    env = PondEnv()
    state_size = env.encode_state().shape[0]
    action_size = env.num_actions()
    agent = DQNAgent(state_size, action_size)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_mask = env.action_mask
            action = agent.select_action(state, action_mask)

            # Décodage de l'ID d'action en action réelle
            available_actions = env.available_actions()
            action_dict = None
            for act in available_actions:
                if env.get_action_id(act) == action:
                    action_dict = act
                    break

            if action_dict is None:
                # Si aucune action correspondante n'est trouvée, choisir aléatoirement
                action_dict = random.choice(available_actions)

            next_state, reward, done = env.step(action_dict, play_random_after_agent=True)
            total_reward += reward

            next_action_mask = env.action_mask
            agent.optimize_model(state, action, reward, next_state, done, next_action_mask)

            state = next_state

        print(f"Épisode {episode+1}/{num_episodes}, Récompense totale: {total_reward}, Epsilon: {agent.epsilon:.4f}")
