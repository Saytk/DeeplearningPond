import time
from tqdm import tqdm
from PondEnv import PondEnv
import random  # Ajout de l'import manquant


def calculate_games_per_second(num_games=100000):
    start_time = time.time()

    for _ in tqdm(range(num_games), desc="Generating games"):
        env = PondEnv()
        env.reset()

        done = False
        while not done:
            available_actions = env.available_actions()
            if len(available_actions) == 0:
                break
            action = random.choice(available_actions)
            _, _, done = env.step(action)

    total_time = time.time() - start_time
    games_per_second = num_games / total_time
    print(f"Games per second: {games_per_second:.2f}")


if __name__ == "__main__":
    calculate_games_per_second(num_games=1000)
