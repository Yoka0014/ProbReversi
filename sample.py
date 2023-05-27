from prob_reversi import Position, Move, DiscColor
from game import IPlayer

class MyPlayer(IPlayer):
    def __init__(self):
        self.__pos = Position(4)

    @property
    def name(self):
        return "MyPlayer"
    
    def set_position(self, pos: Position):
        self.__pos = pos

    def gen_move(self) ->int:
        pos = self.__pos
        moves = pos.get