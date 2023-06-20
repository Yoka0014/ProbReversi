import numpy as np

from game import IPlayer
from prob_reversi import Move, Position
from dualnet import DualNetwork, NN_NUM_CHANNEL, position_to_input

class DualNetPlayer(IPlayer):
    def __init__(self, model_path: str, greedy_value=False):
        self.__pos = Position(4)
        self.__dual_net = DualNetwork(model_path=model_path)
        self.__greedy_value = greedy_value
    
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

        if not self.__greedy_value:
            print(f"win_rate: {(v[0][0] + 1.0) * 50.0:.2f}%")
            return max(moves, key=lambda c: p[0][c])
        else:
            next_positions = [pos.copy() for i in range(len(moves) * 2)]
            batch = np.empty(shape=(len(next_positions), pos.SIZE, pos.SIZE, NN_NUM_CHANNEL))
            for i, move_coord in enumerate(moves):
                idx = i * 2
                p = next_positions[idx] 
                p.do_move(p.get_player_move(move_coord))
                position_to_input(p, batch[idx])

                idx += 1
                p = next_positions[idx] 
                p.do_move(p.get_opponent_move(move_coord))
                position_to_input(p, batch[idx])
            
            v = self.__dual_net.predict(batch, batch_size=len(next_positions))[1]

            values = []
            for i, move_coord in enumerate(moves):
                idx = i * 2
                prob = pos.TRANS_PROB[move_coord]
                values.append(prob * v[idx] + (1.0 - prob) * v[idx + 1])
            
            i, value = min(enumerate(values), key=lambda x: x[1])
            print(f"win_rate: {(-value[0] + 1.0) * 50.0:.2f}%")
            return moves[i]
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()
