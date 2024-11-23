from Token import Token

class Player:
    def __init__(self, color):
        self.color = color
        self.spawn = [Token(color) for _ in range(13)]
        self.score = 0

    def place_piece_from_spawn(self):
        if self.spawn:
            return self.spawn.pop(0)
        else:
            raise ValueError(f"{self.color.capitalize()} has no more pieces to place.")

    def add_to_score(self, count):
        self.score += count

    def remaining_tokens(self):
        return len(self.spawn)
