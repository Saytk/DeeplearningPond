from player import Player
class Board:
    def __init__(self):
        self.grid_size = 4
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.light_player = Player('light')
        self.dark_player = Player('dark')
        self.current_player = self.light_player
        self.turn = 'light'
        self.explayer = 'light'
        self.explayerturn = 'light'
        self.turncount = 0
        self.game_over = False
        self.score_differential = 0
        self.winner = None  # Ajout : 'light', 'dark', ou 'tie'


    def place_piece(self, row, col, piece):
        if self.grid[row][col] is None:
            self.grid[row][col] = piece
            return True
        return False

    def is_adjacent(self, row, col, new_row, new_col):
        return (row == new_row or col == new_col) and abs(row - new_row) + abs(col - new_col) == 1

    def is_valid_frog_move(self, row, col, new_row, new_col):

        if row != new_row and col != new_col:
            return False

        distance = abs(row - new_row) + abs(col - new_col)
        return distance <= 2

    def waitBeforeDevelop(self, timer = 1):
        import time
        time.sleep(timer)
        return True
    def develop_pieces(self, row, col, wait = False):
        if(wait and self.turn == 'light'):
            self.waitBeforeDevelop()
        for d_row, d_col in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_row, adj_col = row + d_row, col + d_col
            if 0 <= adj_row < self.grid_size and 0 <= adj_col < self.grid_size:
                piece = self.grid[adj_row][adj_col]
                if piece:
                    piece.develop()

    def check_for_sets(self):
        scored_light = 0
        scored_dark = 0

        # Horizontal and vertical checking
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Check horizontal sets
                if j <= self.grid_size - 3:
                    length, light, dark = self.get_line_length(i, j, 0, 1)
                    if length >= 3:
                        scored_light += light
                        scored_dark += dark
                        self.clear_line(i, j, 0, 1, length)

                # Check vertical sets
                if i <= self.grid_size - 3:
                    length, light, dark = self.get_line_length(i, j, 1, 0)
                    if length >= 3:
                        scored_light += light
                        scored_dark += dark
                        self.clear_line(i, j, 1, 0, length)

        # Add scored pieces to player scores
        self.light_player.add_to_score(scored_light)
        self.dark_player.add_to_score(scored_dark)
        self.score_differential = self.light_player.score - self.dark_player.score


    def get_line_length(self, row, col, d_row, d_col):
        """Get the length of a line of identical pieces."""
        piece = self.grid[row][col]
        if piece is None:
            return 0, 0, 0

        length = 1
        light_score = 0
        dark_score = 0

        if piece.color == 'light':
            light_score += 1
        else:
            dark_score += 1

        for k in range(1, self.grid_size):
            new_row = row + k * d_row
            new_col = col + k * d_col
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                break
            next_piece = self.grid[new_row][new_col]
            if next_piece is None or next_piece.get_piece_type() != piece.get_piece_type():
                break
            length += 1
            if next_piece.color == 'light':
                light_score += 1
            else:
                dark_score += 1

        return length, light_score, dark_score

    def clear_line(self, row, col, d_row, d_col, length):
        """Clear a line of pieces."""
        for k in range(length):
            self.grid[row + k * d_row][col + k * d_col] = None

    def check_line(self, row, col, d_row, d_col):
        piece = self.grid[row][col]
        if piece is None:
            return False
        for k in range(1, 3):
            if self.grid[row + k * d_row][col + k * d_col] is None or \
               self.grid[row + k * d_row][col + k * d_col].get_piece_type() != piece.get_piece_type():
                return False
        return True

    def score_line(self, row, col, d_row, d_col, scored_light, scored_dark):
        for k in range(3):
            piece = self.grid[row + k * d_row][col + k * d_col]
            if piece.color == 'light':
                scored_light += 1
            else:
                scored_dark += 1
            self.grid[row + k * d_row][col + k * d_col] = None
        return scored_light, scored_dark

    def switch_turn(self):
        self.explayer = self.current_player
        self.explayerturn = self.turn
        self.current_player = self.dark_player if self.current_player == self.light_player else self.light_player
        self.turn = 'dark' if self.turn == 'light' else 'light'
        self.turncount += 1

    def move_piece(self, start_row, start_col, end_row, end_col):
        self.grid[end_row][end_col] = self.grid[start_row][start_col]
        self.grid[start_row][start_col] = None

    def handle_post_move(self, row, col):
        self.develop_pieces(row, col)
        self.check_for_sets()
        self.check_victory_conditions()
        self.switch_turn()
    def check_victory_conditions(self):
        if self.light_player.score >= 10 or self.dark_player.score >= 10:
            self.handle_scoring_victory()

    def handle_scoring_victory(self):
        light_score = self.light_player.score
        dark_score = self.dark_player.score

        if(light_score > dark_score):
            self.winner = 'light'
            #print(f"Light wins with a score of {light_score} to {dark_score}!")
        elif(dark_score > light_score):
            self.winner = 'dark'
           # print(f"Dark wins with a score of {dark_score} to {light_score}!")
        else:
            self.winner = 'tie'
            # print(f"The game ends in a tie with a score of {light_score} to {dark_score}!")
        self.game_over = True

    def handle_elimination(self):
        self.winner = 'dark' if self.current_player.color == 'light' else 'light'
        #print(f"{self.current_player.color.capitalize()} cannot make any moves and loses by elimination!")
        self.game_over = True
    def can_player_make_move(self, player):

        if player.remaining_tokens() > 0:
            return True

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                piece = self.grid[row][col]
                if piece and piece.color == player.color:
                    if piece.get_piece_type() == "Tadpole":
                        if self.can_tadpole_move(row, col):
                            return True
                    elif piece.get_piece_type() == "Frog":
                        if self.can_frog_move(row, col):
                            return True

        return False

    def can_tadpole_move(self, row, col):
        for d_row, d_col in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_row, adj_col = row + d_row, col + d_col
            if 0 <= adj_row < self.grid_size and 0 <= adj_col < self.grid_size:
                if self.grid[adj_row][adj_col] is None:
                    return True
        return False

    def can_frog_move(self, row, col):
        for d_row, d_col in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for steps in range(1, 3):  # Frogs can move up to 2 steps
                new_row, new_col = row + d_row * steps, col + d_col * steps
                if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                    if self.grid[new_row][new_col] is None:
                        return True
        return False


