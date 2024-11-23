from random import choice  # Utiliser `choice` pour les actions aléatoires
from PondEnv import PondEnv

import pygame
from GUI import GameDisplay

# Pygame Visuals

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 550))  # Augmentation de la hauteur pour la grille complète
pygame.display.set_caption("Pond Game")
clock = pygame.time.Clock()

# Colors and Font
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_COLOR = (200, 200, 255)
DARK_COLOR = (100, 100, 255)
HIGHLIGHT_COLOR = (255, 255, 0)  # Jaune pour mettre en surbrillance une pièce
font = pygame.font.SysFont(None, 40)

# Définir le dictionnaire de couleurs
colors = {
    'WHITE': WHITE,
    'BLACK': BLACK,
    'LIGHT_COLOR': LIGHT_COLOR,
    'DARK_COLOR': DARK_COLOR,
    'HIGHLIGHT_COLOR': HIGHLIGHT_COLOR
}

# Initialisation de l'environnement de jeu
env = PondEnv()
state = env.reset()

# Initialisation de l'affichage du jeu
game_display = GameDisplay(screen, env.board, font, colors)

# Configuration du joueur humain (choisissez "light" ou "dark")
human_player = "light"  # Remplacez par "dark" pour jouer le joueur sombre
ai_player = "dark" if human_player == "light" else "light"

# Délai pour le coup de l'IA en millisecondes
ai_delay = 3000  # 1 seconde
ai_last_move_time = 0  # Dernier temps où l'IA a joué

# Boucle principale du jeu
running = True
while running:
    current_time = pygame.time.get_ticks()  # Temps écoulé depuis le démarrage du jeu

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # L'utilisateur agit uniquement si c'est son tour
        if env.board.turn == human_player and not env.board.game_over:
            game_display.handle_input(event)

    # Si c'est au tour de l'adversaire IA
    if env.board.turn == ai_player and not env.board.game_over:
        # Vérifier si le délai est écoulé avant de jouer le coup
        if current_time - ai_last_move_time >= ai_delay:
            available_actions = env.available_actions()
            if available_actions:
                random_action = choice(available_actions)
                state, reward, done = env.step(random_action, play_random_after_agent=False)
                ai_last_move_time = current_time  # Mettre à jour le temps du dernier coup de l'IA

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
