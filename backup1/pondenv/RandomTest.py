import pygame
import random
from PondEnv import PondEnv
from GUI import GameDisplay

pygame.init()
screen = pygame.display.set_mode((400, 550))
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 40)
colors = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'LIGHT_COLOR': (200, 200, 255),
    'DARK_COLOR': (100, 100, 255),
    'HIGHLIGHT_COLOR': (255, 255, 0)
}

env = PondEnv()
game_display = GameDisplay(screen, env.board, font, colors)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    available_actions = env.available_actions()

    if len(available_actions) == 0:
        print("Game Over: Aucun mouvement disponible.")
        running = False
        break

    action = random.choice(available_actions)
    new_state, reward, done = env.step(action)

    screen.fill((255, 255, 255))

    game_display.draw_score()
    game_display.draw_grid()
    game_display.draw_pieces()
    pygame.display.flip()

    clock.tick(1)

    if done:
        print("Game Over!")
        running = False
