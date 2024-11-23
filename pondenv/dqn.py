import numpy as np
import torch
from base_dqn import BaseDQN
from model_utils import save_model


class DQN(BaseDQN):
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

            self.decay_epsilon()

            if episode % eval_interval == 0:
                winrate, lose_rate, tie_rate = evaluate(self.env, self.q_network)
                print(f"Episode {episode} | Win Rate: {winrate:.2f}%")
                model_path = save_model(self.q_network, winrate)
                best_models = update_best_models(winrate, model_path, best_models, top_k=5)

        return best_models
