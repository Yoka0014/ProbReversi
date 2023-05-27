"""
モンテカルロ法で1手先の局面を評価して着手を決定するプレイヤー.
"""
import random

from game import IPlayer
from prob_reversi import DiscColor, Move, Position

class MCPlayer(IPlayer):
    def __init__(self, playout_num=100):
        self.__pos = Position(4)
        self.__playout_num = playout_num

    @property
    def name(self):
        return "MC Player"
    
    def set_position(self, pos: Position):
        self.__pos = pos
    
    def gen_move(self) -> int:
        side_to_move = self.__pos.side_to_move
        moves = list(self.__pos.get_next_moves())
        values = []
        for move in moves:
            pos = self.__pos.copy()
            pos.do_move_at(move)
            value = 0.0
            p = pos.copy()
            for _ in range(self.__playout_num):
                value += self.__playout(p, side_to_move)
                pos.copy_to(p)
            value /= self.__playout_num
            values.append(value)
        return moves[max(enumerate(values), key=lambda x: x[1])[0]]

    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()

    def __playout(self, pos: Position, color: DiscColor) -> float:
        pass_count = 0
        while pass_count != 2:
            coord = pos.sample_next_move()
            if coord == pos.PASS_COORD:
                pos.do_pass()
                pass_count += 1
                continue

            pass_count = 0    
            move = pos.get_move(coord)
            pos.do_move(move)

        score = pos.get_score_from(color)

        if score == 0:
            return 0.5
        
        return 1.0 if score > 0 else 0.0
        
        