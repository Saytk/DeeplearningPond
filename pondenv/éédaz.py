import numpy as np
import random
from Board import Board


class PondEnv:
    def __init__(self):
        self.board = Board()
        self.current_state_id = 0
        self.reset()

    def reset(self):
        self.board = Board()
        self.current_state_id = self.encode_state()
        #print("Game has been reset.")
        self.display()
        return self.current_state_id

    def display(self):
        #print("Current Board State:")
        for row in self.board.grid:
            pass
            #print([str(piece) if piece is not None else "None" for piece in row])
        #print()

    def num_states(self) -> int:
        return self.board.grid_size ** 2 * 3

    def num_actions(self) -> int:
        return len(self.generate_actions_for_current_player())

    def reward(self) -> float:
        return self.board.light_player.score + self.board.dark_player.score  # Total score as the reward

    def step(self, action: dict):
        #print(f"Taking action: {action}")
        if action["type"] == "place":
            piece = self.board.current_player.place_piece_from_spawn()
            self.board.place_piece(action["row"], action["col"], piece)
            self.board.handle_post_move(action["row"], action["col"])
        elif action["type"] == "move":
            self.board.move_piece(action["start_row"], action["start_col"], action["end_row"], action["end_col"])
            self.board.handle_post_move(action["end_row"], action["end_col"])

        self.current_state_id = self.encode_state()
        #print("New state after action:")
        self.display()
        return self.current_state_id, self.reward(), self.is_game_over()

    def available_actions(self) -> list:
        return self.generate_actions_for_current_player()

    def encode_state(self) -> np.ndarray:
        state_vector = np.zeros((self.board.grid_size ** 2) * 4 + 5, dtype=int)

        for row in range(self.board.grid_size):
            for col in range(self.board.grid_size):
                piece = self.board.grid[row][col]
                start_idx = (row * self.board.grid_size + col) * 4
                if piece is None:
                    state_vector[start_idx:start_idx + 4] = [0, 0, 0, 0]  # Aucune pièce
                else:
                    encoded_token = piece.encode()
                    state_vector[start_idx:start_idx + 4] = encoded_token

        state_vector[-5] = self.board.light_player.score
        state_vector[-4] = self.board.dark_player.score

        state_vector[-3] = self.board.light_player.remaining_tokens()
        state_vector[-2] = self.board.dark_player.remaining_tokens()

        state_vector[-1] = 1 if self.board.turn == 'light' else 0

        return state_vector

    def encode_action(action: dict, grid_size: int = 4) -> list:
        """
        Encode une action sur un vecteur de 144 entiers représentant toutes les actions possibles
        sur une grille de 4x4, avec 9 actions potentielles par case.
        """
        # Initialiser un vecteur de 144 entiers à 0
        encoded_action = [0] * (grid_size * grid_size * 9)

        if action["type"] == "place":
            # Encodage de l'action "place" (ID 0)
            idx = action["row"] * grid_size + action["col"]
            encoded_action[idx * 9] = 1  # Premier entier pour "place"

        elif action["type"] == "move":
            # Encodage de l'action "move"
            start_row = action["start_row"]
            start_col = action["start_col"]
            end_row = action["end_row"]
            end_col = action["end_col"]

            start_idx = start_row * grid_size + start_col
            vertical_delta = end_row - start_row
            horizontal_delta = end_col - start_col

            if vertical_delta == -1 and horizontal_delta == 0:
                # Déplacement d'une case vers le haut (ID 1)
                encoded_action[start_idx * 9 + 1] = 1
            elif vertical_delta == 0 and horizontal_delta == 1:
                # Déplacement d'une case vers la droite (ID 2)
                encoded_action[start_idx * 9 + 2] = 1
            elif vertical_delta == 1 and horizontal_delta == 0:
                # Déplacement d'une case vers le bas (ID 3)
                encoded_action[start_idx * 9 + 3] = 1
            elif vertical_delta == 0 and horizontal_delta == -1:
                # Déplacement d'une case vers la gauche (ID 4)
                encoded_action[start_idx * 9 + 4] = 1
            elif vertical_delta == -2 and horizontal_delta == 0:
                # Déplacement de deux cases vers le haut (ID 5)
                encoded_action[start_idx * 9 + 5] = 1
            elif vertical_delta == 0 and horizontal_delta == 2:
                # Déplacement de deux cases vers la droite (ID 6)
                encoded_action[start_idx * 9 + 6] = 1
            elif vertical_delta == 2 and horizontal_delta == 0:
                # Déplacement de deux cases vers le bas (ID 7)
                encoded_action[start_idx * 9 + 7] = 1
            elif vertical_delta == 0 and horizontal_delta == -2:
                # Déplacement de deux cases vers la gauche (ID 8)
                encoded_action[start_idx * 9 + 8] = 1

        return encoded_action

    def generate_actions_for_current_player(self):
        actions = []
        current_player = self.board.current_player.color
        has_remaining_tokens = self.board.current_player.remaining_tokens() > 0

        action_generators = {
            'Tadpole': self.generate_tadpole_moves,
            'Frog': self.generate_frog_moves
        }

        for row in range(self.board.grid_size):
            for col in range(self.board.grid_size):
                piece = self.board.grid[row][col]

                if piece is None and has_remaining_tokens:
                    actions.append(self.create_placement_action(row, col))

                elif piece is not None and piece.color == current_player:
                    move_generator = action_generators.get(piece.get_piece_type())
                    if move_generator:
                        actions += move_generator(row, col)

        return actions

    def create_placement_action(self, row, col):
        return {
            "type": "place",
            "row": row,
            "col": col,
            "piece": "Egg"
        }

    def generate_tadpole_moves(self, row, col):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for d_row, d_col in directions:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < self.board.grid_size and 0 <= new_col < self.board.grid_size:
                if self.board.grid[new_row][new_col] is None:
                    moves.append({
                        "type": "move",
                        "start_row": row,
                        "start_col": col,
                        "end_row": new_row,
                        "end_col": new_col
                    })
        return moves

    def generate_frog_moves(self, row, col):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for d_row, d_col in directions:
            for step in range(1, 3):
                new_row, new_col = row + d_row * step, col + d_col * step
                if 0 <= new_row < self.board.grid_size and 0 <= new_col < self.board.grid_size:
                    if self.board.grid[new_row][new_col] is None:
                        moves.append({
                            "type": "move",
                            "start_row": row,
                            "start_col": col,
                            "end_row": new_row,
                            "end_col": new_col
                        })
        return moves

    def is_game_over(self) -> bool:
        return self.board.game_over


# Simple test script
if __name__ == "__main__":
    env = PondEnv()
    env.reset()

    #print("\nAvailable actions:", env.available_actions())

    for step in range(500):
        available_actions = env.available_actions()
        if len(available_actions) == 0:
            #print("No more available actions. Game Over.")
            break

        action = random.choice(available_actions)
        new_state, reward, done = env.step(action)
        #print(f"Step {step}: State={new_state}, Reward={reward}, Done={done}")

        if done:
            #print("Game Over!")
            break
