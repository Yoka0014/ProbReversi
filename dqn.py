"""DQNを用いて確率リバーシを学習するためのモジュール

* このファイルには，DQNで確率リバーシを学習するのに必要なデータ構造やクラスが定義されている.
"""
import random

import numpy as np
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential, load_model, clone_model
from keras.optimizers import Adam

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
        dest = np.zeros(shape=(2, pos.SIZE, pos.SIZE))
    elif dest.shape != (2, pos.SIZE, pos.SIZE):
        raise ValueError("The shape of dest must be (2, pos.SIZE, pos.SIZE).")

    for coord in pos.get_player_disc_coords():
        dest[0].ravel()[coord] = 1

    for coord in pos.get_opponent_disc_coords():
        dest[1].ravel()[coord] = 1

    return dest


class QNetwork:
    """
    Qネットワーク

    先手の石の配置と後手の石の配置を (2, board_size, board_size) の三次元データとして入力し，各マス目の行動価値を board_size * board_size 次元のベクトルとして出力するNN．
    """

    def __init__(self, board_size=6, kernel_num=128, model_path: str = None, src=None):
        if model_path is not None:
            self.__model = load_model(model_path)
        elif src is not None:
            self.__BOARD_SIZE = src.__BOARD_SIZE
            self.__KERNEL_NUM = src.__KERNEL_NUM
            self.__model = clone_model(src.__model)
        else:
            self.__BOARD_SIZE = board_size
            self.__KERNEL_NUM = kernel_num
            self.__model = Sequential()
            self.__init_model()

    def __init_model(self):
        size = self.__BOARD_SIZE
        k = self.__KERNEL_NUM
        self.__model = Sequential()
        model = self.__model

        model.add(Conv2D(k, (3, 3), padding="same", input_shape=(2, size, size), data_format="channels_first"))
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
        model.add(Dense(size ** 2, activation="tanh"))

    def save(self, path: str):
        self.__model.save(path)

    def copy(self):
        return QNetwork(src=self)

    def compile(self, optimizer=Adam(lr=0.001), loss="huber_loss", metrics=['mse']):
        self.__model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, x: np.ndarray, batch_size=32) -> np.ndarray:
        """
        NNへの入力を受け取って，その出力を返す．

        この関数では，(batch_size, board_size, board_size, 2)のテンソルをNNに入力して，
        (batch_size, board_size * board_size)の行列として出力を得る.

        Parameters
        ----------
            x: NNへの入力
        """
        return self.__model.predict(x, batch_size=batch_size)

    def predict_from_position(self, pos: Position) -> np.ndarray:
        """
        盤面を受け取って，その盤面における各マスの行動価値を出力する.
        """
        size = self.__BOARD_SIZE
        if pos.SIZE != size:
            raise ValueError("pos.SIZE does not match the input shape of NN.")

        x = np.zeros(shape=(2, size, size))
        position_to_input(pos, x)
        return self.predict(x)


class TDTarget:
    """
    行動価値

    Attributes
    ----------
    move: Move
        着手．

    value: float
        TDターゲットの値.
    """

    def __init__(self, move: Move, value: float):
        self.move: Move = move
        self.value: float = value

    def to_nn_target(self, board_size: int, dest: np.ndarray = None):
        """
        行動価値をNNのターゲットに変換する.
        """
        if dest is None:
            dest = np.zeros(shape=(board_size))
        elif dest.shape != (board_size):
            raise ValueError("The shape of dest must be (board_size).")

        dest[self.move.coord] = self.value


class Episode:
    """
    1エピソード分の情報を管理するクラス.
    """

    def __init__(self, root_pos: Position):
        self.__root_pos = root_pos  # 初期局面
        self.__history: list[TDTarget] = []  # 着手及びその価値の履歴.

    def len(self):
        return len(self.__history)

    def add_experience(self, action_value: TDTarget):
        self.__history.append(action_value)

    def sample_state(self) -> tuple[Position, TDTarget]:
        """
        エピソードの中から，状態(盤面)を1つだけサンプリングし，状態とTDターゲットのペアを返す.
        """
        i = random.randint(0, len(self.__history) - 1)
        pos = self.__root_pos.copy()
        for entry in self.__history[0:i]:
            pos.do_move(entry[0])
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
        self.__episodes: list[Episode] = []
        self.__window_size: int = window_size
        self.__board_size = board_size

    @property
    def window_size(self) -> int:
        return self.__window_size

    @property
    def board_size(self) -> int:
        return self.__board_size
    
    def __len__(self):
        return len(self.__episodes)

    def save_episode(self, episode: Episode):
        if len(self.__episodes) >= self.__window_size:
            self.__episodes.pop(0)
        self.__episodes.append(episode)

    def sample_batch(self, batch_size: int, dest: tuple[np.ndarray, np.ndarray]=None) -> tuple[np.ndarray, np.ndarray]:
        """
        引数で与えられた数だけ，ランダムにエピソードを選び，それぞれのエピソードの中から，状態を1つランダムサンプリングして，NN学習用のバッチを作る.
        """
        if dest is None:
            dest = np.zeros(shape=(batch_size, 2, self.__board_size, self.__board_size))
        elif dest[0].shape != (batch_size, 2, self.__board_size, self.__board_size):
            raise ValueError("The shape of dest[0] must be (batch_size, 2, board_size, board_size).")
        elif dest[1].shape != (batch_size, self.__board_size * self.__board_size):
            raise ValueError("The shape of dest[1] must be (batch_size, board_size * board_size).")

        episode_len_sum = float(sum(e.len()) for e in self.__episodes)
        episodes = np.random.choice(self.__episodes, size=batch_size,
                                    p=[e.len() / episode_len_sum for e in self.__episodes])

        inputs, targets = dest
        for i, (s, q) in enumerate(map(lambda e: e.sample_state(), episodes)):
            position_to_input(s, inputs[i])
            q.to_nn_target(targets[i])
