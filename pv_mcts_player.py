from game import IPlayer
from prob_reversi import Move, Position
from pv_mcts import SearchResult, MoveEval, UCTConfig, UCT

class PV_MCTSPlayer(IPlayer):
    def __init__(self, config: UCTConfig, num_playout, num_retries=5):
        self.__pos = Position(4)
        self.__uct = UCT(config)
        self.__num_playout = num_playout
        self.__num_retries = num_retries
    
    @property
    def name(self):
        return "PV-MCTS Player"

    def set_position(self, pos: Position):
        self.__pos = pos
        self.__uct.set_root_pos(pos)
    
    def gen_move(self) -> int:
        moves = list(self.__pos.get_next_moves()) 
        if len(moves) == 1:
            return moves[0]

        retry_count = 0
        while True:
            result = self.__uct.search(self.__num_playout)
            move_evals = sorted(result.move_evals, key=lambda e: 1.0 - e.effort)
            best_value_move = max(move_evals, key=lambda e: e.action_value)

            print(self.__uct.get_search_result_str())

            if move_evals[0] == best_value_move and move_evals[0].effort > move_evals[1].effort * 1.5:
                break

            if retry_count <= self.__num_retries:
                retry_count += 1
                print(f"\nEnter panic time: {retry_count}")
            else:
                break

        return move_evals[0].coord
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)
        self.__uct.set_root_pos(self.__pos)

    def do_pass(self):
        self.__pos.do_pass()
        self.__uct.set_root_pos(self.__pos)
