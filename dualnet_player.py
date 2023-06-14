from game import IPlayer
from prob_reversi import Move, Position
from dualnet import DualNetwork

class DualNetPlayer(IPlayer):
    def __init__(self, model_path: str):
        self.__pos = Position(4)
        self.__dual_net = DualNetwork(model_path=model_path)
    
    @property
    def name(self):
        return "DualNet Player"

    def set_position(self, pos: Position):
        self.__pos = pos
    
    def gen_move(self) -> int:
        pos = self.__pos
        moves = list(pos.get_next_moves())
        if len(moves) == 0:
            return pos.PASS_COORD
        p, v = self.__dual_net.predict_from_position(pos)
        print(f"win_rate: {(v[0][0] + 1.0) * 50.0:.2f}%")
        return max(moves, key=lambda c: p[0][c])
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()
