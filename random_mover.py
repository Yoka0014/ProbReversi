"""
乱数で着手を決定するプレイヤー.
"""
from game import IPlayer
from prob_reversi import Move, Position

class RandomMover(IPlayer):
    def __init__(self):
        self.__pos = Position(4)

    @property
    def name(self):
        return "RandomMover"
    
    def set_position(self, pos: Position):
        self.__pos = pos
    
    def gen_move(self) -> int:
        return self.__pos.sample_next_move()
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()