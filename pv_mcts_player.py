from game import IPlayer
from prob_reversi import Move, Position
from pv_mcts import SearchResult, MoveEval, UCTConfig, UCT

class PV_MCTSPlayer(IPlayer):
    def __init__(self, config: UCTConfig, num_playout):
        self.__pos = Position(4)
        self.__uct = UCT(config)
        self.__num_playout = num_playout
    
    @property
    def name(self):
        return "PV-MCTS Player"

    def set_position(self, pos: Position):
        self.__pos = pos
        self.__uct.set_root_pos(pos)
    
    def gen_move(self) -> int:
        result = self.__uct.search(self.__num_playout)
        best = max(result.move_evals, key=lambda e: e.playout_count)
        print(self.__uct.get_search_result_str())
        return best.coord
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)
        self.__uct.set_root_pos(self.__pos)

    def do_pass(self):
        self.__pos.do_pass()
        self.__uct.set_root_pos(self.__pos)
