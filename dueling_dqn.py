"""DQNを用いて確率リバーシを学習するためのモジュール

* このファイルには，DQNで確率リバーシを学習するのに必要なデータ構造やクラスが定義されている.
"""
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Add
from keras.models import Model, load_model, clone_model
from keras.optimizers import Adam
from keras.losses import Huber
from keras.activations import relu
from keras.initializers.initializers_v2 import HeNormal, GlorotNormal 
from keras.regularizers import L2

from prob_reversi import Position, Move, DiscColor


"""
NNに入力するデータのチャンネル数.

channel = 1: 現在の手番の石の配置(2値画像)
channel = 2: 相手の石の配置(2値画像)
"""
NN_NUM_CHANNEL = 2


def position_to_input(pos: Position, dest: np.ndarray = None) -> np.ndarray:
    """
    受け取ったPositionオブジェクトをNNへの入力に変換する.

    Parameters
    ----------
    pos: Position
        局面．

    dest: np.ndarray
        書き込み先のndarray. Noneの場合は関数内で新たに作る.

    Returns
    -------
    dest: np.ndarray
        引数のdestと同じ参照.
    """
    if dest is None:
        dest = np.empty(shape=(pos.SIZE, pos.SIZE, NN_NUM_CHANNEL))
    elif dest.shape != (pos.SIZE, pos.SIZE, NN_NUM_CHANNEL):
        raise ValueError("The shape of dest must be (pos.SIZE, pos.SIZE, 2).")

    dest.fill(0.0)
    for coord in pos.get_player_disc_coords():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][0] = 1.0

    for coord in pos.get_opponent_disc_coords():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][1] = 1.0

    return dest


class QNetwork:
    """
    Qネットワーク

    先手の石の配置と後手の石の配置を (board_size, board_size, NN_NUM_CAHNNEL) の三次元データとして入力し，
    各マス目の行動価値を board_size * board_size + 1 次元のベクトルとして出力するNN．
    """

    # L2正則化の定数
    L2 = 5.0e-4

    def __init__(self, board_size=6, num_kernel=192, num_res_block=10,
                 model_path: str = None, src=None, optimizer=Adam(lr=0.001), loss=Huber()):
        """
        コンストラクタ

        Parameters
        ----------
        board_size: int
            盤面のサイズ(デフォルトは6)

        kernel_num: int
            カーネル(フィルター)数(デフォルトは192)

        num_res_block: int
            ResBlockの数(デフォルトは10)

        model_path: str
            ファイルからパラメータをロードする場合のファイルパス(Noneなら無視される)

        src: QNetwork
            別のQNetworkオブジェクトからパラメータをコピーする際のコピー元(Noneなら無視される)

        optimizer: tf.Optimizer
            学習の際に用いるオプティマイザ(デフォルトはAdam)

        loss: tf.Loss
            学習の際に用いる損失関数(デフォルトはHuber関数)
        """
        if model_path is not None:
            # ファイルが指定されていれば，そこからモデルを読む．
            self.__model = load_model(model_path)
        elif src is not None and type(src) is QNetwork:
            # 引数でコピー元のQNetworkが指定されていれば，そのパラメータをself.__modelにコピーする．
            self.__BOARD_SIZE = src.__model.input_shape[0]
            self.__model = clone_model(src.__model)
            self.__model.set_weights(src.__model.get_weights())
        else:
            # 引数で指定された盤面サイズとカーネル数に基づいてモデルを構築する.
            self.__BOARD_SIZE = board_size
            self.__NUM_KERNEL = num_kernel
            self.__NUM_RES_BLOCK = num_res_block
            self.__model: Model = None
            self.__init_model()

        # 引数で指定されたオプティマイザと損失関数をモデルに登録.
        self.__model.compile(optimizer=optimizer, loss=loss)

    def __del__(self):
        del self.__model

    def __init_model(self):
        """
        モデルを初期化する．
        """
        board_size = self.__BOARD_SIZE

        def conv(num_kernel=self.__NUM_KERNEL, kernel_size=3):
            return Conv2D(num_kernel, kernel_size, padding='same', use_bias=False, kernel_initializer=HeNormal(), kernel_regularizer=L2(QNetwork.L2))

        input = Input(shape=(board_size, board_size, NN_NUM_CHANNEL))
        x = conv()(input)
        x = BatchNormalization()(x)
        x = relu(x)

        for _ in range(self.__NUM_RES_BLOCK):
            sc = x
            x = conv()(x)
            x = BatchNormalization()(x)
            x = relu(x)
            x = conv()(x)
            x = BatchNormalization()(x)
            x = Add()([x, sc])
            x = relu(x)

        a = conv(2, 1)(x)
        a = BatchNormalization()(a)
        a = relu(a)
        a = Flatten()(a)
        advantages = Dense(board_size ** 2 + 1, kernel_initializer=GlorotNormal(), kernel_regularizer=L2(QNetwork.L2))(a)

        v = conv(1, 1)(x)
        v = BatchNormalization()(v)
        v = relu(v)
        v = Dense(256, kernel_initializer=HeNormal(), kernel_regularizer=L2(QNetwork.L2))(v)
        v = relu(v)
        v = Flatten()(v)
        value = Dense(1, kernel_initializer=GlorotNormal(), kernel_regularizer=L2(QNetwork.L2))(v)

        self.__model = Model(inputs=input, outputs=[advantages, value])


    def save(self, path: str):
        """
        モデルをファイルに保存する.

        Parameters
        ----------
        path: str
            モデルの保存先のファイルパス
        """
        self.__model.save(path)

    def copy(self):
        """
        QNetworkオブジェクトをパラメータごと複製したオブジェクトを返す(深いコピー).

        Returns
        -------
        copied: QNetwork
        """
        return QNetwork(src=self)

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return self.__model.predict(x, verbose=0)
    
    def predict_q(self, x: np.ndarray) -> np.ndarray:
        a, v = self.predict_raw(x)
        return tf.nn.tanh(a + v - np.mean(a, axis=1, keepdims=True)).numpy()

    def predict_q_from_position(self, pos: Position) -> np.ndarray:
        size = pos.SIZE
        x = np.zeros(shape=(1, size, size, NN_NUM_CHANNEL))
        position_to_input(pos, x[0])
        return self.predict_q(x)
    
    def predict_vq_from_position(self, pos: Position) -> np.ndarray:
        size = pos.SIZE
        x = np.zeros(shape=(1, size, size, NN_NUM_CHANNEL))
        position_to_input(pos, x[0])
        a, v = self.predict_raw(x)
        return tf.nn.tanh(v - np.mean(a, axis=1, keepdims=True)).numpy(), tf.nn.tanh(a + v - np.mean(a, axis=1, keepdims=True)).numpy()

    def train(self, x: np.ndarray, moves: list[Move], td_targets: np.ndarray, terminal_rewards: np.ndarray) -> float:
        model = self.__model

        # 着手に対応する要素のみ1とするone-hotベクトルをmovesに格納されている着手の数だけ生成する.
        masks = tf.one_hot(list(map(lambda m: m.coord, moves)), self.__BOARD_SIZE ** 2 + 1)
        with tf.GradientTape() as tape:
            advantages, value = model(x)

            # NNが出力したアドバンテージのうち，実際に行った着手に対応するもののみを残す.
            advantage = tf.reduce_sum(advantages * masks, axis=1, keepdims=True)

            q = tf.nn.tanh(advantage + value - tf.reduce_mean(advantages, axis=1, keepdims=True))

            # 損失を求める.
            loss = (model.loss(td_targets, q) + model.loss(terminal_rewards, q)) * 0.5

        grads = tape.gradient(loss, model.trainable_weights)
        clipped_grads = []
        for g in grads:
            clipped_grads.append(tf.clip_by_value(g, -1.0, 1.0))
        model.optimizer.apply_gradients(zip(clipped_grads, model.trainable_weights))
        return loss


class Episode:
    """
    1エピソード分の情報を管理するクラス.
    """

    def __init__(self, root_pos: Position):
        """
        コンストラクタ

        Parameters
        ----------
        root_pos: Position
            初期局面
        """
        self.__root_pos = root_pos.copy()  # 初期局面
        self.__history: list[Move] = []  # 着手履歴.
        self.__terminal_reward_from_black: float = None

    def set_terminal_reward(self, color: DiscColor, reward: float):
        self.__terminal_reward_from_black = reward if color == DiscColor.BLACK else -reward

    def get_terminal_reward(self, color: DiscColor):
        reward = self.__terminal_reward_from_black
        return reward if color == DiscColor.BLACK else -reward

    def len(self):
        return len(self.__history)

    def add_move(self, move: Move):
        self.__history.append(move)

    def sample_state(self) -> tuple[Position, Move]:
        """
        エピソードの中から，状態(局面)を1つだけサンプリングし，(局面，着手)のペアを返す.
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
    capacity: int
        保存するエピソードの最大数．エピソード数がこの値を超えると最も古いエピソードが削除される.
    """

    def __init__(self, capacity: int, board_size: int):
        self.__episodes: deque[Episode] = deque(maxlen=capacity)
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
        引数で与えられた数だけランダムにエピソードを選び，それぞれのエピソードの中から状態を1つランダムサンプリングして，NN学習用のバッチを作る.

        Returns
        -------
        batch: list[(Position, Move, Position, float)]
            (局面，着手，着手後の局面，終端報酬(着手前の手番から見た), next_posが終局かどうか)    
            ただし，報酬は終局時にのみ得られる
        """
        episode_len_sum = float(sum(e.len() for e in self.__episodes))
        episodes = np.random.choice(self.__episodes, size=batch_size,
                                    p=[e.len() / episode_len_sum for e in self.__episodes])

        batch = []
        for episode in episodes:
            episode: Episode
            pos, move = episode.sample_state()

            next_pos = pos.copy()
            is_terminal = False
            if move.coord == pos.PASS_COORD:
                next_pos.do_pass()
                if next_pos.can_pass():  # パスを2連続でできるなら終局
                    is_terminal = True
            else:
                next_pos.do_move(move)

            batch.append((pos, move, next_pos, episode.get_terminal_reward(pos.side_to_move), is_terminal))

        return batch
