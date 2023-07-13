"""DQNを用いて確率リバーシを学習するためのモジュール

* このファイルには，DQNで確率リバーシを学習するのに必要なデータ構造やクラスが定義されている.
"""
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten
from keras.losses import Huber
from keras.models import Sequential, clone_model, load_model
from keras.optimizers import Adam

from prob_reversi import Move, Position

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

    def __init__(self, board_size=6, kernel_num=128, layer_num=5, 
                 model_path: str = None, src=None, copy_optimizer=False, copy_loss=False, optimizer=Adam(lr=0.001), loss=Huber()):
        """
        コンストラクタ

        Parameters
        ----------
        board_size: int
            盤面のサイズ(デフォルトは6)

        kernel_num: int
            カーネル(フィルター)数(デフォルトは128)

        layer_num: int
            NNの層数(デフォルトは5)

        model_path: str
            ファイルからパラメータをロードする場合のファイルパス(Noneなら無視される)

        src: QNetwork
            別のQNetworkオブジェクトからパラメータをコピーする際のコピー元(Noneなら無視される)

        copy_optimizer: bool
            srcからオプティマイザもコピーするかどうか

        copy_loss: bool
            srcから損失関数もコピーするかどうか

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
            if copy_optimizer:
                optimizer = src.__model.optimizer
            if copy_loss:
                loss = src.__model.loss
        else:
            # 引数で指定された盤面サイズとカーネル数に基づいてモデルを構築する.
            self.__BOARD_SIZE = board_size
            self.__KERNEL_NUM = kernel_num
            self.__LAYER_NUM = layer_num
            self.__model = Sequential()
            self.__init_model()

        # 引数で指定されたオプティマイザと損失関数をモデルに登録.
        self.__model.compile(optimizer=optimizer, loss=loss)

    def __del__(self):
        del self.__model

    def __init_model(self):
        """
        モデルを初期化する．
        """
        size = self.__BOARD_SIZE
        k = self.__KERNEL_NUM
        self.__model = Sequential()
        model = self.__model

        for _ in range(self.__LAYER_NUM - 1):
            # (層数 - 1)回だけ 畳み込み->バッチ正規化->ReLU関数 の流れを繰り返す(最後の1層は出力層)
            model.add(Conv2D(k, (3, 3), padding="same", input_shape=(size, size, NN_NUM_CHANNEL), use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(size ** 2 + 1, activation="tanh"))

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

    def predict(self, x: np.ndarray, batch_size=32) -> np.ndarray:
        """
        QNetworkへの入力のバッチを受け取って，その出力を返す．

        この関数では，(batch_size, board_size, board_size, NUM_NN_CHANNEL)のテンソルをNNに入力して，
        (batch_size, board_size ** 2 + 1)の行列として出力を得る.

        Parameters
        ----------
            x: np.ndarray[shape=(batch_size, board_size, board_size, NN_NUM_CHANNEL)]
                QNetworkへの入力のバッチ

        Returns
        -------
        np.ndarray[shape=(batch_size, board_size ** 2 + 1)]
            QNetworkの出力のバッチ
        """
        return self.__model.predict(x, batch_size=batch_size, verbose=0)

    def predict_from_position(self, pos: Position) -> np.ndarray:
        """
        局面を受け取って，その盤面における各マスの行動価値を出力する.

        Parameters
        ----------
        pos: Position
            各着手の行動価値を算出する局面

        Returns
        -------
        np.ndarray[shape=(1, board_size ** 2 + 1)]
            各マス目とパスの行動価値
        """
        size = pos.SIZE
        x = np.zeros(shape=(1, size, size, NN_NUM_CHANNEL))
        position_to_input(pos, x[0])
        return self.predict(x, batch_size=1)

    def train(self, x: np.ndarray, td_targets: np.ndarray, moves: list[Move]) -> float:
        """
        与えられたバッチから勾配計算を1回行って重みを更新する.

        Parameters
        ----------
        x: np.ndarray[shape=(batch_size, board_size, board_size, NN_NUM_CHANNEL)]
            QNetworkへの入力のバッチ

        td_targets: np.ndarray[shape=(batch_size, 1)]
            TDターゲットのバッチ

        moves: list[Move]
            バッチ内の各盤面で行った着手

        Returns
        -------
        loss: np.ndarray
            損失関数の出力
        """
        model = self.__model

        # 着手に対応する要素のみ1とするone-hotベクトルをmovesに格納されている着手の数だけ生成する.
        masks = tf.one_hot(list(map(lambda m: m.coord, moves)), self.__BOARD_SIZE ** 2 + 1)
        with tf.GradientTape() as tape:
            # 損失の計算をGradientTapeに記録する.

            # NNで各着手の行動価値を出力.
            q = model(x)

            # NNが出力した行動価値のうち，実際に行った着手の行動価値のみを残して他を0にする.
            # Note: tf.math.multiplyは要素ごとの積なので，masksと乗算することでqの不要な要素を0にできる.
            q = tf.reduce_sum(masks * q, axis=1, keepdims=True)

            # 損失を求める.
            loss = model.loss(td_targets, q)

        # lossをNNのパラメータで微分して勾配ベクトルを求める.
        grads = tape.gradient(loss, model.trainable_weights)

        # 求めた勾配ベクトルでNNのパラメータを更新する.
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
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
            (局面，着手，着手後の局面，報酬)    
            ただし，報酬は終局時にのみ得られる
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
                if next_pos.can_pass():  # パスを2連続でできるなら終局
                    score = next_pos.get_score()
                    if score == 0:
                        reward = 0.0
                    else:
                        reward = 1.0 if score > 0 else -1.0
            else:
                next_pos.do_move(move)

            batch.append((pos, move, next_pos, reward))

        return batch
