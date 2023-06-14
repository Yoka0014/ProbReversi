from game import IPlayer
from prob_reversi import Move, Position
from mcts import SearchResult, MoveEval, UCTConfig, UCT

class MCTSPlayer(IPlayer):
    def __init__(self, config: UCTConfig, num_playout=1000):
        self.__pos = Position(4)
        self.__uct = UCT(config)
        self.__num_playout = num_playout
    
    @property
    def name(self):
        return "MCTS Player"

    def set_position(self, pos: Position):
        self.__pos = pos
        self.__uct.set_root_pos(pos)
    
    def gen_move(self) -> int:
        result = self.__uct.search(self.__num_playout)
        best = max(result.move_values, key=lambda e: e.playout_count)
        print(f"win_rate: {best.action_value * 100.0:.2f}%")
        return best.coord
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)
        self.__uct.set_root_pos(self.__pos)

    def do_pass(self):
        self.__pos.do_pass()
        self.__uct.set_root_pos(self.__pos)
