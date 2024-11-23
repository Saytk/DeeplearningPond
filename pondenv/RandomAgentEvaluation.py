import random
import numpy as np
from PondEnv import PondEnv

# Define Random Agent
class RandomAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self):
        actions = self.env.generate_actions_for_current_player()
        if not actions:
            return None
        return random.choice(actions)

# Evaluate Random Agent as Light Player
def evaluate_random_agent(env, num_eval_episodes=1000):
    results = {'win': 0, 'lose': 0, 'tie': 0}

    for _ in range(num_eval_episodes):
        env.reset()  # Reset the environment
        done = False

        while not done:
            # Light (Random Agent) plays first
            if env.board.turn == 'light':
                random_action = env.generate_actions_for_current_player()
                if random_action:
                    action = random.choice(random_action)
                    env.step(action, play_random_after_agent=False)

            # Dark (Random Agent)
            if not env.board.game_over and env.board.turn == 'dark':
                random_action = env.generate_actions_for_current_player()
                if random_action:
                    action = random.choice(random_action)
                    env.step(action, play_random_after_agent=False)

            done = env.board.game_over

        # Record the result
        if env.board.winner == 'light':
            results['win'] += 1
        elif env.board.winner == 'dark':
            results['lose'] += 1
        else:
            results['tie'] += 1

    total_games = sum(results.values())
    winrate = (results['win'] / total_games) * 100
    lose_rate = (results['lose'] / total_games) * 100
    tie_rate = (results['tie'] / total_games) * 100

    print("\n--- Results for Random Agent ---")
    print(f"- Win Rate (Light): {winrate:.2f}%")
    print(f"- Lose Rate (Dark): {lose_rate:.2f}%")
    print(f"- Tie Rate: {tie_rate:.2f}%")

    return results

# Main
if __name__ == "__main__":
    env = PondEnv()

    # Evaluate Random Agent
    evaluate_random_agent(env, num_eval_episodes=100000)
