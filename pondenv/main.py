import pygame
from Token import Token
from GUI import GameDisplay
from Board import Board

# Pygame Visuals

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 550))  # Increased height for full grid
pygame.display.set_caption("Pond Game")
clock = pygame.time.Clock()

# Colors and Font
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_COLOR = (200, 200, 255)
DARK_COLOR = (100, 100, 255)
HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow for selected piece
font = pygame.font.SysFont(None, 40)

# Define the color dictionary
colors = {
    'WHITE': WHITE,
    'BLACK': BLACK,
    'LIGHT_COLOR': LIGHT_COLOR,
    'DARK_COLOR': DARK_COLOR,
    'HIGHLIGHT_COLOR': HIGHLIGHT_COLOR
}

# Create and initialize the Board
board = Board()

# Add 3 Frogs and 1 Egg on the first row
board.grid[0][0] = Token('light')  # Frog
board.grid[0][0].develop()  # Develop to Tadpole
board.grid[0][0].develop()  # Develop to Frog

board.grid[0][1] = Token('light')  # Frog
board.grid[0][1].develop()
board.grid[0][1].develop()

board.grid[0][2] = Token('light')  # Frog
board.grid[0][2].develop()
board.grid[0][2].develop()

board.grid[0][3] = Token('light')  # Egg (unchanged)
board.grid[0][3].develop()

# Initialize GameDisplay
game_display = GameDisplay(screen, board, font, colors)

# Main Game Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        game_display.handle_input(event)

    screen.fill(WHITE)
    game_display.draw_score()
    game_display.draw_grid()
    game_display.draw_pieces()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
