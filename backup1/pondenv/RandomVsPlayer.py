import pygame
import random
import numpy as np
from PondEnv import PondEnv
from DQNAgent import QNetwork, select_action
import torch
import torch.nn as nn

class HumanVsAgentGame:
    def __init__(self, screen, env, agent, font, colors, device):
        self.screen = screen
        self.env = env
        self.agent = agent
        self.font = font
        self.colors = colors
        self.device = device
        self.running = True
        self.selected_piece = None
        self.clock = pygame.time.Clock()

    def draw_board(self):
        """Dessine la grille, les pièces et les scores."""
        self.screen.fill(self.colors['WHITE'])
        self.draw_grid()
        self.draw_pieces()
        self.draw_score()
        pygame.display.flip()

    def draw_grid(self):
        """Dessine la grille de jeu."""
        for x in range(0, 400, 100):
            for y in range(130, 530, 100):
                rect = pygame.Rect(x, y, 100, 100)
                pygame.draw.rect(self.screen, self.colors['BLACK'], rect, 1)

    def draw_pieces(self):
        """Dessine les pièces sur la grille."""
        for row in range(self.env.board.grid_size):
            for col in range(self.env.board.grid_size):
                piece = self.env.board.grid[row][col]
                if piece:
                    color = self.colors['LIGHT_COLOR'] if piece.color == 'light' else self.colors['DARK_COLOR']
                    center_x = col * 100 + 50
                    center_y = row * 100 + 180

                    pygame.draw.circle(self.screen, color, (center_x, center_y), 30)

    def draw_score(self):
        """Affiche le score et l'état du jeu."""
        light_score_text = self.font.render(f"Light: {self.env.board.light_player.score}", True, self.colors['LIGHT_COLOR'])
        dark_score_text = self.font.render(f"Dark: {self.env.board.dark_player.score}", True, self.colors['DARK_COLOR'])
        turn_text = self.font.render(f"Turn: {self.env.board.turn.capitalize()}", True, self.colors['BLACK'])
        self.screen.blit(light_score_text, (10, 10))
        self.screen.blit(dark_score_text, (220, 10))
        self.screen.blit(turn_text, (150, 50))

    def human_turn(self, event):
        """Gérer le tour du joueur humain."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if y >= 130 and not self.env.board.game_over:
                row, col = (y - 130) // 100, x // 100
                available_actions = self.env.generate_actions_for_current_player()

                for action in available_actions:
                    if action["type"] == "place" and action["row"] == row and action["col"] == col:
                        self.env.step(action)
                        return True
                    elif action["type"] == "move" and action["end_row"] == row and action["end_col"] == col:
                        self.env.step(action)
                        return True
        return False

    def agent_turn(self):
        """Laisser l'agent jouer son tour."""
        state = torch.tensor(self.env.encode_state(), dtype=torch.float32).to(self.device)
        self.env.generate_actions_for_current_player()
        action_mask = self.env.action_mask
        q_values = self.agent(state)

        try:
            action_idx = select_action(q_values, action_mask, epsilon=0.0)
        except ValueError:
            print("[DEBUG] Aucun mouvement valide pour l'agent.")
            return False

        action = self.env.decode_action(action_idx)
        self.env.step(action)
        return True

    def run(self):
        """Boucle principale du jeu."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if self.env.board.turn == 'dark':
                    if self.human_turn(event):
                        self.env.board.switch_turn()

            if self.env.board.turn == 'light' and not self.env.board.game_over:
                if self.agent_turn():
                    self.env.board.switch_turn()

            self.draw_board()
            self.clock.tick(30)

        pygame.quit()


# Lancer le jeu
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((400, 550))
    pygame.display.set_caption("Human vs Agent")
    font = pygame.font.SysFont(None, 40)

    colors = {
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'LIGHT_COLOR': (200, 200, 255),
        'DARK_COLOR': (100, 100, 255),
        'HIGHLIGHT_COLOR': (255, 255, 0)
    }

    env = PondEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(env.num_states(), env.num_actions()).to(device)
    q_network.load_state_dict(torch.load("best_q_network.pth"))  # Chargez le modèle pré-entraîné

    game = HumanVsAgentGame(screen, env, q_network, font, colors, device)
    game.run()
