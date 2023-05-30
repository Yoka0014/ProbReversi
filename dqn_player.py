from game import IPlayer
from prob_reversi import Move, Position
import dqn

class DQNPlayer(IPlayer):
    def __init__(self, model_path: str):
        self.__pos = Position(4)
        self.__qnet = dqn.QNetwork(model_path=model_path)
    
    @property
    def name(self):
        return "DQN Player"

    def set_position(self, pos: Position):
        self.__pos = pos
    
    def gen_move(self) -> int:
        pos = self.__pos
        moves = list(pos.get_next_moves())
        if len(moves) == 0:
            return pos.PASS_COORD
        q = self.__qnet.predict_from_position(pos)[0]
        return max(moves, key=lambda c: q[c])
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()
