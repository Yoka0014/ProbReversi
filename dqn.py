"""DQNを用いて確率リバーシを学習するためのモジュール

* このファイルには，DQNで確率リバーシを学習するのに必要なデータ構造やクラスが定義されている.
"""
import random
from collections import deque
from typing import Any

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential, load_model, clone_model
from keras.optimizers import Adam
from keras.losses import Huber

from prob_reversi import Position, Move


def position_to_input(pos: Position, dest: np.ndarray = None) -> np.ndarray:
    """
    受け取ったPositionオブジェクトをNNへの入力に変換する.

    Parameters
    ----------
    pos: Position
        盤面．

    dest: np.ndarray
        書き込み先のndarray. Noneの場合は関数内で新たに作る.

    Returns
    -------
    dest: np.ndarray
        引数のdestと同じ参照.
    """
    if dest is None:
        dest = np.empty(shape=(pos.SIZE, pos.SIZE, 2))
    elif dest.shape != (pos.SIZE, pos.SIZE, 2):
        raise ValueError("The shape of dest must be (pos.SIZE, pos.SIZE, 2).")

    dest.fill(0.0)
    for coord in pos.get_player_disc_coords():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][0] = 1

    for coord in pos.get_opponent_disc_coords():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][1] = 1

    return dest


class QNetwork:
    """
    Qネットワーク

    先手の石の配置と後手の石の配置を (2, board_size, board_size) の三次元データとして入力し，各マス目の行動価値を board_size * board_size + 1 次元のベクトルとして出力するNN．
    """

    def __init__(self, board_size=6, kernel_num=128, model_path: str = None, optimizer=Adam(lr=0.001), loss=Huber()):
        if model_path is not None:
            self.__model = load_model(model_path)
        else:
            self.__BOARD_SIZE = board_size
            self.__KERNEL_NUM = kernel_num
            self.__model = Sequential()
            self.__init_model()

        self.__optimizer = optimizer
        self.__loss_func = loss

    def __init_model(self):
        size = self.__BOARD_SIZE
        k = self.__KERNEL_NUM
        self.__model = Sequential()
        model = self.__model

        model.add(Conv2D(k, (3, 3), padding="same", input_shape=(size, size, 2)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(k, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(k, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(k, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(size ** 2 + 1, activation="tanh"))

    def save(self, path: str):
        self.__model.save(path)

    def copy(self):
        return QNetwork(src=self)

    def predict(self, x: np.ndarray, batch_size=32) -> np.ndarray:
        """
        NNへの入力を受け取って，その出力を返す．

        この関数では，(batch_size, board_size, board_size, 2)のテンソルをNNに入力して，
        (batch_size, board_size * board_size)の行列として出力を得る.

        Parameters
        ----------
            x: NNへの入力
        """
        return self.__model.predict(x, batch_size=batch_size, verbose=0)

    def predict_from_position(self, pos: Position) -> np.ndarray:
        """
        盤面を受け取って，その盤面における各マスの行動価値を出力する.
        """
        size = pos.SIZE
        x = np.zeros(shape=(1, size, size, 2))
        position_to_input(pos, x[0])
        return self.predict(x, batch_size=1)
    
    def train(self, x, y, moves: list[Move]) -> float:
        """
        与えられたバッチから，勾配計算を1回行って重みを更新する.

        Parameters
        ----------
        x: np.ndarray
            入力のバッチ
        
        y: np.ndarray
            出力のバッチ

        moves: list[Move]
            学習対象の着手

        Returns
        -------
        loss: float
            損失関数の値
        """
        model = self.__model
        masks = tf.one_hot(list(map(lambda m: m.coord, moves)), self.__BOARD_SIZE ** 2 + 1)
        with tf.GradientTape() as tape:
            y_preds = model(x)
            y_preds = tf.math.multiply(y_preds, masks)
            loss = self.__loss_func(y, y_preds)

        grads = tape.gradient(loss, model.trainable_weights)
        self.__optimizer.apply_gradients(zip(grads, model.trainable_weights))



class Episode:
    """
    1エピソード分の情報を管理するクラス.
    """

    def __init__(self, root_pos: Position):
        self.__root_pos = root_pos.copy()  # 初期局面
        self.__history: list[Move] = []  # 着手履歴.

    def len(self):
        return len(self.__history)

    def add_move(self, move: Move):
        self.__history.append(move)

    def sample_state(self) -> tuple[Position, Move]:
        """
        エピソードの中から，状態(盤面)を1つだけサンプリングし，(盤面，着手)のペアを返す.
        """
        i = random.randint(0, len(self.__history) - 1)
        pos = self.__root_pos.copy()
        for move in self.__history[0:i]:
            if move.coord == pos.PASS_COORD:
                pos.do_pass()
            else:
                pos.do_move(move)

        return pos, self.__history[i]
        


class ReplayBuffer:
    """
    経験再生用のバッファ.
    今までのエピソードを保管する.

    Attributes
    ----------
    window_size: int
        保存するエピソードの最大数．エピソード数がこの値を超えると，最も古いエピソードが削除される.
    """

    def __init__(self, window_size: int, board_size: int):
        self.__episodes: deque[Episode] = deque(maxlen=window_size)
        self.__board_size = board_size

    @property
    def window_size(self) -> int:
        return self.__episodes.maxlen

    @property
    def board_size(self) -> int:
        return self.__board_size
    
    def __len__(self):
        return len(self.__episodes)

    def save_episode(self, episode: Episode):
        self.__episodes.append(episode)

    def sample_batch(self, batch_size: int) -> list[tuple[Position, Move, Position, float]]:
        """
        引数で与えられた数だけ，ランダムにエピソードを選び，それぞれのエピソードの中から，状態を1つランダムサンプリングして，NN学習用のバッチを作る.

        Returns
        -------
        batch: list[(Position, Move, Position, float)]
            (局面，着手，着手後の局面，報酬)    
            ただし，報酬は終局時のみ得られる
        """
        episode_len_sum = float(sum(e.len() for e in self.__episodes))
        episodes = np.random.choice(self.__episodes, size=batch_size,
                                    p=[e.len() / episode_len_sum for e in self.__episodes])
        
        batch = []
        for pos, move in map(lambda e: e.sample_state(), episodes):
            pos: Position
            move: Move

            next_pos = pos.copy()
            reward = None
            if move.coord == pos.PASS_COORD:
                next_pos.do_pass()
                if next_pos.can_pass(): # パスを2連続でできるなら終局
                    score = next_pos.get_score()
                    if score == 0:
                        reward = 0.0
                    else:
                        reward = 1.0 if score > 0 else -1.0 
            else:
                next_pos.do_move(move)

            batch.append((pos, move, next_pos, reward))

        return batch
                


            






            
