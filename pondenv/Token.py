class Token:
    EGG = 0
    TADPOLE = 1
    FROG = 2
    piece_types = ['Egg', 'Tadpole', 'Frog']

    def __init__(self, color):
        self.type = Token.EGG
        self.color = color

    def develop(self):
        self.type = (self.type + 1) % 3

    def get_piece_type(self):
        return Token.piece_types[self.type]

    def encode(self):
        encoded_token = [0] * 4

        if self.color == 'light':
            encoded_token[0] = 1
        else:
            encoded_token[0] = 0

        if self.type == Token.EGG:
            encoded_token[1:] = [0, 0, 1]
        elif self.type == Token.TADPOLE:
            encoded_token[1:] = [0, 1, 0]
        elif self.type == Token.FROG:
            encoded_token[1:] = [1, 0, 0]

        return encoded_token

    def __str__(self):
        color_str = 'Light' if self.color == 'light' else 'Dark'
        type_str = self.get_piece_type()
        return f'{color_str} {type_str}'


token = Token('light')
print(token)
token.develop()
print(token)
