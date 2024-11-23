import pygame
from Board import Board

class GameDisplay:
    def __init__(self, screen, board, font, colors):
        self.screen = screen
        self.board = board
        self.font = font
        self.colors = colors
        self.selected_piece = None

    def draw_grid(self):
        for x in range(0, 400, 100):
            for y in range(130, 530, 100):
                rect = pygame.Rect(x, y, 100, 100)
                pygame.draw.rect(self.screen, self.colors['BLACK'], rect, 1)

    def draw_pieces(self):
        for row in range(self.board.grid_size):
            for col in range(self.board.grid_size):
                piece = self.board.grid[row][col]
                if piece:
                    color = self.colors['LIGHT_COLOR'] if piece.color == 'light' else self.colors['DARK_COLOR']
                    center_x = col * 100 + 50
                    center_y = row * 100 + 180

                    if self.selected_piece == (row, col):
                        pygame.draw.circle(self.screen, self.colors['HIGHLIGHT_COLOR'], (center_x, center_y), 40)  # Highlight

                    pygame.draw.circle(self.screen, color, (center_x, center_y), 30)

                    piece_type = piece.get_piece_type()
                    symbol = piece_type[0]
                    text = self.font.render(symbol, True, self.colors['BLACK'])
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.screen.blit(text, text_rect)

    def draw_score(self):
        light_score_text = self.font.render(f"Light: {self.board.light_player.score}", True, self.colors['LIGHT_COLOR'])
        dark_score_text = self.font.render(f"Dark: {self.board.dark_player.score}", True, self.colors['DARK_COLOR'])

        light_tokens_text = self.font.render(f"Remaining: {self.board.light_player.remaining_tokens()}", True, self.colors['LIGHT_COLOR'])
        dark_tokens_text = self.font.render(f"Remaining: {self.board.dark_player.remaining_tokens()}", True, self.colors['DARK_COLOR'])

        turn_text = self.font.render(f"Turn: {self.board.turn.capitalize()}", True, self.colors['BLACK'])

        self.screen.blit(light_score_text, (10, 10))
        self.screen.blit(dark_score_text, (220, 10))
        self.screen.blit(light_tokens_text, (10, 50))
        self.screen.blit(dark_tokens_text, (220, 50))
        self.screen.blit(turn_text, (150, 110))

    def handle_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()

            if y >= 130 and not self.board.game_over:
                row, col = (y - 130) // 100, x // 100

                if 0 <= row < self.board.grid_size and 0 <= col < self.board.grid_size:
                    if self.selected_piece is None:
                        if self.board.grid[row][col] is not None:
                            if self.board.grid[row][col].color == self.board.current_player.color:
                                self.selected_piece = (row, col)
                        else:
                            try:
                                piece = self.board.current_player.place_piece_from_spawn()
                                if self.board.place_piece(row, col, piece):
                                    self.board.handle_post_move(row, col)
                            except ValueError:
                                pass

                    else:
                        piece = self.board.grid[self.selected_piece[0]][self.selected_piece[1]]
                        if piece.get_piece_type() == "Tadpole":
                            if self.board.is_adjacent(self.selected_piece[0], self.selected_piece[1], row, col) and self.board.grid[row][col] is None:
                                self.board.move_piece(self.selected_piece[0], self.selected_piece[1], row, col)
                                self.selected_piece = None
                                self.board.handle_post_move(row, col)
                        elif piece.get_piece_type() == "Frog":
                            if self.board.is_valid_frog_move(self.selected_piece[0], self.selected_piece[1], row, col) and self.board.grid[row][col] is None:
                                self.board.move_piece(self.selected_piece[0], self.selected_piece[1], row, col)
                                self.selected_piece = None
                                self.board.handle_post_move(row, col)
                        else:
                            self.selected_piece = None
