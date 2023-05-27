from game import IPlayer
from prob_reversi import Move, Position

class MyPlayer(IPlayer):
    def __init__(self):
        self.__pos = Position(4)
    
    @property
    def name(self):
        return "My Player"

    def set_position(self, pos: Position):
        self.__pos = pos
    
    def gen_move(self) -> int:
        pos = self.__pos
        moves = pos.get_next_moves()
        return max(moves, lambda c: pos.TRANS_PROB[c])
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()
