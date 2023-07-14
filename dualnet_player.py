import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from game import IPlayer
from prob_reversi import Move, Position
from dualnet import position_to_input, DN_NUM_CHANNEL

class DualNetPlayer(IPlayer):
    def __init__(self, board_size, model_path: str, max_policy=True):
        self.__pos = Position(board_size)
        self.__dual_net: Model = load_model(model_path)
        self.__max_policy = max_policy
    
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
        
        x = np.empty(shape=(1, pos.SIZE, pos.SIZE, DN_NUM_CHANNEL))
        position_to_input(pos, x[0])
        p_logits, v = self.__dual_net.predict(x, verbose=0)

        if self.__max_policy:
            move = max(moves, key=lambda x: p_logits[0][x])
        else:
            policy = p_logits[0][moves].numpy()
            policy = tf.nn.softmax(policy, axis=1).numpy().astype(np.float64)
            policy /= np.sum(policy)
            move = np.random.choice(moves, p=policy).item()

        print(f"win_rate: {(v[0].item() + 1.0) * 50:.2f}%")

        return move
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()
