import numpy as np
import torch
from random import choice
from PondEnv import PondEnv
from DQNAgent import QNetwork, select_action
from GUI import GameDisplay
import pygame

# Charger le modèle DQN
def load_dqn_model(model_path, state_dim, action_dim):
    q_network = QNetwork(state_dim, action_dim)
    q_network.load_state_dict(torch.load(model_path))
    q_network.eval()
    return q_network

# Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode((400, 550))
pygame.display.set_caption("Pond Game - Human vs DQN")
clock = pygame.time.Clock()

# Définition des couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_COLOR = (200, 200, 255)
DARK_COLOR = (100, 100, 255)
HIGHLIGHT_COLOR = (255, 255, 0)
font = pygame.font.SysFont(None, 40)
colors = {
    'WHITE': WHITE,
    'BLACK': BLACK,
    'LIGHT_COLOR': LIGHT_COLOR,
    'DARK_COLOR': DARK_COLOR,
    'HIGHLIGHT_COLOR': HIGHLIGHT_COLOR
}

# Initialisation de l'environnement et de l'affichage
env = PondEnv()
state_dim = len(env.reset())
action_dim = env.num_actions()
game_display = GameDisplay(screen, env.board, font, colors)

# Charger le modèle DQN
dqn_model_path = 'q_network_87.90.pth'
dqn_agent = load_dqn_model(dqn_model_path, state_dim, action_dim)

# Configurer les joueurs
human_player = "dark"  # L'humain contrôle les pièces sombres
dqn_player = "light"  # Le modèle DQN contrôle les pièces claires

# Délai pour le coup de l'IA
ai_delay = 3000  # 1 seconde
ai_last_move_time = 0

# Boucle principale du jeu
running = True
while running:
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Si c'est au tour de l'humain
        if env.board.turn == human_player and not env.board.game_over:
            game_display.handle_input(event)
            ai_last_move_time = current_time  # Mettre à jour le délai


    # Si c'est au tour du DQN agent
    if env.board.turn == dqn_player and not env.board.game_over:
        if current_time - ai_last_move_time >= ai_delay:
            # Générer les actions valides
            env.generate_actions_for_current_player()
            action_mask = env.action_mask

            if np.any(action_mask):
                # Sélectionner une action avec le DQN
                state = torch.tensor(env.encode_state(), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = dqn_agent(state).squeeze(0)
                action_idx = select_action(q_values, action_mask, epsilon=0.0)
                action = env.decode_action(action_idx)

                # Exécuter l'action
                state, reward, done = env.step(action, play_random_after_agent=False)

                if done:
                    print(f"Game Over! Winner: {env.board.winner}")
                    running = False

    # Mise à jour graphique
    screen.fill(WHITE)
    game_display.draw_score()
    game_display.draw_grid()
    game_display.draw_pieces()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

