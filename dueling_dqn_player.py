from game import IPlayer
from prob_reversi import Move, Position
import dueling_dqn

class DuelingDQNPlayer(IPlayer):
    def __init__(self, model_path: str):
        self.__pos = Position(4)
        self.__qnet = dueling_dqn.QNetwork(model_path=model_path)
    
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
        v, q = self.__qnet.predict_vq_from_position(pos)
        best_move = max(moves, key=lambda c: q[0][c])
        print(f"\nwin_rate(v): {(v[0].item() + 1.0) * 50.0:.2f}\n")
        print(f"win_rate(q): {(q[0][best_move].item() + 1.0) * 50.0:.2f}\n")
        return best_move
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()
