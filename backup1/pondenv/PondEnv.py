import numpy as np
import random
from Board import Board


class PondEnv:
    def __init__(self):
        self.board = Board()
        self.current_state_id = 0
        self.reset()
        self.action_mask = np.zeros(144)


    def reset(self):
        self.board = Board()
        self.current_state_id = self.encode_state()
        return self.current_state_id

    def display(self):
        print("Current Board State:")
        for row in self.board.grid:
            print([str(piece) if piece is not None else "None" for piece in row])
        print()

    def num_states(self) -> int:
        return 69

    def num_actions(self) -> int:
        return 144

    def reward(self) -> float:
        light_score = self.board.light_player.score
        dark_score = self.board.dark_player.score
        current_score_differential = light_score - dark_score
        reward = current_score_differential - self.board.score_differential
        if not self.board.game_over:
            return 0.1 if reward > 0 else 0

        if self.board.winner == 'light':
            return reward + 1  # Victoire pour light
        elif self.board.winner == 'dark':
            return -1  # Défaite pour light
        elif self.board.winner == 'tie':
            return 0  # Match nul

    def step(self, action: dict, play_random_after_agent: bool = True):
        """
        Exécute une action pour l'agent (always "light"), et fait jouer un adversaire aléatoire si activé.

        :param action: L'action à exécuter pour l'agent (light).
        :param play_random_after_agent: Si True, l'adversaire (dark) joue un coup aléatoire après l'action de l'agent.
        :return: Un tuple (current_state_id, reward, done).
        """
        if action["type"] == "place":
            piece = self.board.current_player.place_piece_from_spawn()
            self.board.place_piece(action["row"], action["col"], piece)
            self.board.handle_post_move(action["row"], action["col"])
        elif action["type"] == "move":
            self.board.move_piece(action["start_row"], action["start_col"], action["end_row"], action["end_col"])
            self.board.handle_post_move(action["end_row"], action["end_col"])

        if self.board.game_over:
            return self.encode_state(), self.reward(), True

        if play_random_after_agent and self.board.turn == "dark":
            available_actions = self.available_actions()
            if available_actions:
                random_action = random.choice(available_actions)
                if random_action["type"] == "place":
                    piece = self.board.current_player.place_piece_from_spawn()
                    self.board.place_piece(random_action["row"], random_action["col"], piece)
                    self.board.handle_post_move(random_action["row"], random_action["col"])
                elif random_action["type"] == "move":
                    self.board.move_piece(
                        random_action["start_row"],
                        random_action["start_col"],
                        random_action["end_row"],
                        random_action["end_col"]
                    )
                    self.board.handle_post_move(random_action["end_row"], random_action["end_col"])

            if self.board.game_over:
                return self.encode_state(), self.reward(), True

        self.current_state_id = self.encode_state()
        return self.current_state_id, self.reward(), self.board.game_over

    def available_actions(self) -> list:
        return self.generate_actions_for_current_player()

    def encode_state(self) -> np.ndarray:
        state_vector = np.zeros((self.board.grid_size ** 2) * 4 + 5, dtype=int)

        for row in range(self.board.grid_size):
            for col in range(self.board.grid_size):
                piece = self.board.grid[row][col]
                start_idx = (row * self.board.grid_size + col) * 4
                if piece is None:
                    state_vector[start_idx:start_idx + 4] = [0, 0, 0, 0]
                else:
                    encoded_token = piece.encode()
                    state_vector[start_idx:start_idx + 4] = encoded_token

        state_vector[-5] = self.board.light_player.score
        state_vector[-4] = self.board.dark_player.score
        state_vector[-3] = self.board.light_player.remaining_tokens()
        state_vector[-2] = self.board.dark_player.remaining_tokens()
        state_vector[-1] = 1 if self.board.turn == 'light' else 0

        return state_vector

    def get_action_id(self, action: dict, grid_size: int = 4) -> int:
        if action["type"] == "place":
            return (action["row"]*grid_size + action["col"]) * 9

        elif action["type"] == "move":
            start_idx = action["start_row"] * grid_size + action["start_col"]
            vertical_delta = action["end_row"] - action["start_row"]
            horizontal_delta = action["end_col"] - action["start_col"]

            if vertical_delta == -1 and horizontal_delta == 0:
                move_type = 1
            elif vertical_delta == 1 and horizontal_delta == 0:
                move_type = 3
            elif vertical_delta == 0 and horizontal_delta == -1:
                move_type = 4
            elif vertical_delta == 0 and horizontal_delta == 1:
                move_type = 2
            elif vertical_delta == -2 and horizontal_delta == 0:
                move_type = 5
            elif vertical_delta == 2 and horizontal_delta == 0:
                move_type = 7
            elif vertical_delta == 0 and horizontal_delta == -2:
                move_type = 8
            elif vertical_delta == 0 and horizontal_delta == 2:
                move_type = 6
            else:
                raise ValueError("Invalid move action parameters.")

            return start_idx * 9 + move_type

        else:
            raise ValueError("Unknown action type.")

    def decode_action(self, action_id: int, grid_size: int = 4) -> dict:
        """
        Décoder un ID d'action en une action compréhensible par l'environnement.

        :param action_id: ID global de l'action (entre 0 et 143).
        :param grid_size: Taille de la grille (par défaut 4x4).
        :return: Un dictionnaire représentant l'action.
        """
        if action_id < 0 or action_id >= grid_size * grid_size * 9:
            raise ValueError(
                f"Action ID {action_id} est hors des limites valides (0 à {grid_size * grid_size * 9 - 1}).")

        # Trouver la case et le type d'action
        cell_id = action_id // 9  # ID de la case (0 à 15 pour une grille 4x4)
        action_type = action_id % 9  # Type d'action (0 à 8)

        # Calculer les coordonnées de la case
        row = cell_id // grid_size
        col = cell_id % grid_size

        if action_type == 0:
            # Placer un oeuf
            return {
                "type": "place",
                "row": row,
                "col": col,
                "piece": "Egg"
            }
        elif 1 <= action_type <= 4:
            # Déplacement d'une case (Têtard)
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Haut, Droite, Bas, Gauche
            d_row, d_col = deltas[action_type - 1]
            return {
                "type": "move",
                "start_row": row,
                "start_col": col,
                "end_row": row + d_row,
                "end_col": col + d_col
            }
        elif 5 <= action_type <= 8:
            # Déplacement de deux cases (Grenouille)
            deltas = [(-2, 0), (0, 2), (2, 0), (0, -2)]  # Haut, Droite, Bas, Gauche
            d_row, d_col = deltas[action_type - 5]
            return {
                "type": "move",
                "start_row": row,
                "start_col": col,
                "end_row": row + d_row,
                "end_col": col + d_col
            }
        else:
            raise ValueError(f"Type d'action {action_type} est invalide.")

    def step_with_encoded_action(self, action_vector: np.ndarray, play_random_after_agent: bool = True):
        """
        Surcharge de la fonction step pour accepter une action encodée sous forme de vecteur.

        :param action_vector: Un vecteur de longueur 144 représentant une action encodée.
        :param play_random_after_agent: Si True, l'adversaire joue aléatoirement après l'agent.
        :return: Un tuple (current_state_id, reward, done).
        """
        if len(action_vector) != self.num_actions():
            raise ValueError(
                f"Le vecteur d'action doit avoir une longueur de {self.num_actions()}, mais {len(action_vector)} a été fourni.")

        # Trouver l'ID de l'action (indice du bit à 1 dans le vecteur)
        action_id = np.argmax(action_vector)

        # Décoder l'action
        decoded_action = self.decode_action(action_id)

        # Appeler la méthode step existante avec l'action décodée

        return self.step(decoded_action, play_random_after_agent=play_random_after_agent)

    def encode_action(self, action: dict, grid_size: int = 4) -> np.ndarray:
        action_id = self.get_action_id(action, grid_size)
        encoded_action = np.zeros(grid_size * grid_size * 9, dtype=int)
        encoded_action[action_id] = 1
        return encoded_action

    def generate_actions_for_current_player(self):
        """
        Génère les actions disponibles pour le joueur actuel.
        Met à jour le masque des actions uniquement si le joueur actuel est 'light'.
        """
        actions = []
        current_player = self.board.current_player.color
        has_remaining_tokens = self.board.current_player.remaining_tokens() > 0

        action_generators = {
            'Tadpole': self.generate_tadpole_moves,
            'Frog': self.generate_frog_moves
        }

        # Si le joueur actuel est 'light', initialiser le masque
        if current_player == 'light':
            self.action_mask = np.zeros(144, dtype=int)

        for row in range(self.board.grid_size):
            for col in range(self.board.grid_size):
                piece = self.board.grid[row][col]

                if piece is None and has_remaining_tokens:
                    action = self.create_placement_action(row, col)
                    actions.append(action)
                    # Mettre à jour le masque uniquement si 'light'
                    if current_player == 'light':
                        idx_action = self.get_action_id(action)
                        self.action_mask[idx_action] = 1

                elif piece is not None and piece.color == current_player:
                    move_generator = action_generators.get(piece.get_piece_type())
                    if move_generator:
                        moves = move_generator(row, col)
                        actions.extend(moves)
                        # Mettre à jour le masque pour chaque mouvement si 'light'
                        if current_player == 'light':
                            for move in moves:
                                self.action_mask[self.get_action_id(move)] = 1
        if actions == []:
            self.board.handle_elimination()
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

    # Simuler un tour avec 'light'
    env.board.current_player.color = 'light'
    actions_light = env.generate_actions_for_current_player()
    print("Actions Light:", actions_light)
    print("Mask Light:", env.action_mask)

    # Simuler un tour avec 'dark'
    env.board.current_player.color = 'dark'
    actions_dark = env.generate_actions_for_current_player()
    print("Actions Dark:", actions_dark)
    print("Mask Dark:", env.action_mask)  # Doit rester vide ou inchangé
