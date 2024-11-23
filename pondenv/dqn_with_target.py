import torch
from base_dqn import BaseDQN
from model_utils import save_model


class DQNWithTarget(BaseDQN):
    def __init__(self, env, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, target_update_freq=100):
        super().__init__(env, state_dim, action_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay)
        self.target_network = self._build_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_update_freq = target_update_freq

    def train_step(self, state, action, reward, next_state, done, action_mask):
        with torch.no_grad():
            target_q = reward
            if not done:
                next_q_values = self.target_network(next_state)
                max_next_q_value = torch.max(next_q_values[action_mask]).item()
                target_q += self.gamma * max_next_q_value

        predicted_q_value = self.q_network(state)[action]
        loss = self.loss_fn(predicted_q_value, torch.tensor(target_q, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, num_episodes, eval_interval=10):
        best_models = []
        for episode in range(num_episodes):
            self.env.reset()
            state = torch.tensor(self.env.encode_state(), dtype=torch.float32)
            done = False

            while not done:
                self.env.generate_actions_for_current_player()
                action_mask = self.env.action_mask
                if not np.any(action_mask):
                    break

                action = self.select_action(state, action_mask)
                action_vector = np.zeros(self.action_dim, dtype=int)
                action_vector[action] = 1

                next_state, reward, done = self.env.step_with_encoded_action(action_vector)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                loss = self.train_step(state, action, reward, next_state, done, torch.tensor(action_mask, dtype=torch.bool))
                state = next_state

            if episode % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            self.decay_epsilon()

            if episode % eval_interval == 0:
                winrate, lose_rate, tie_rate = evaluate(self.env, self.q_network)
                print(f"Episode {episode} | Win Rate: {winrate:.2f}%")
                model_path = save_model(self.q_network, winrate)
                best_models = update_best_models(winrate, model_path, best_models, top_k=5)

        return best_models
