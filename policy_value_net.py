from typing import Any
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Dense, BatchNormalization, Activation, Flatten
from keras.models import Model, load_model, clone_model
from keras.optimizers import Adam
from keras.losses import MSE, CategoricalCrossentropy
from keras.regularizers import L2

from prob_reversi import Position

"""
NNに入力するデータのチャンネル数.

channel = 1: 現在の手番の石の配置(2値画像)
channel = 2: 相手の石の配置(2値画像)
channel = 3: 着手可能位置(2値画像)
channel = 4: 空きマスの着手成功確率
"""
NN_NUM_CHANNEL = 4


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

    for coord in pos.get_next_moves():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][2] = 1.0

    for coord in pos.get_empty_square_coords():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][3] = pos.TRANS_PROB[coord]

    return dest

class PolicyNetwork:
    def __init__(self, board_size=6, num_kernel=192, num_block=9, model_path: str = None, src=None, optimizer=Adam(learning_rate=0.001)):
        if model_path is not None:
            # ファイルが指定されていれば，そこからモデルを読む．
            self.__model = load_model(model_path)
        elif src is PolicyNetwork:
            self.__BOARD_SIZE = src.__model.input_shape[0]
            self.__model = clone_model(src.__model)
            self.__model.set_weights(src.__model.get_weights())
        else:
            # 引数で指定された盤面サイズとカーネル数に基づいてモデルを構築する.
            self.__BOARD_SIZE = board_size
            self.__NUM_KERNEL = num_kernel
            self.__NUM_BLOCK = num_block
            self.__model = self.__init_model()
            self.__model.compile(optimizer=optimizer)

    def call(self, x):
        return self.__model(x)

    def __del__(self):
        del self.__model

    def __init_model(self) -> Model:
        def conv(num_kernel=self.__NUM_KERNEL, kernel_size=3) -> Conv2D:
            return Conv2D(num_kernel, kernel_size, padding="same", use_bias=False,
                          kernel_initializer="he_normal", kernel_regularizer=L2(0.0005))

        def res_block():
            def f(x):
                sc = x
                x = conv()(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                x = conv()(x)
                x = BatchNormalization()(x)
                x = Add()([x, sc])
                x = Activation("relu")(x)
                return x
            return f

        input = Input(shape=(self.__BOARD_SIZE, self.__BOARD_SIZE, NN_NUM_CHANNEL))
        x = conv()(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        for _ in range(self.__NUM_BLOCK):
            x = res_block()(x)

        x = conv(2, 1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        policy = Dense(self.__BOARD_SIZE ** 2 + 1, kernel_initializer="he_normal", kernel_regularizer=L2(0.0005), activation="softmax")(x)

        return Model(inputs=input, outputs=[policy])
    
    @property
    def weights(self):
        return self.__model.trainable_variables
    
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
        PolicyNetworkオブジェクトをパラメータごと複製したオブジェクトを返す(深いコピー).

        Returns
        -------
        copied: PolicyNetwork
        """
        return PolicyNetwork(src=self)

    def predict_logit(self, x: np.ndarray) -> np.ndarray:
        """
        PolicyNetworkへの入力のバッチを受け取って，その出力を返す．
        """
        return self.__model.predict(x, verbose=0)

    def predict_from_position(self, pos: Position) -> np.ndarray:
        size = pos.SIZE
        x = np.zeros(shape=(1, size, size, NN_NUM_CHANNEL))
        position_to_input(pos, x[0])
        return self.predict_logit(x, batch_size=1)
    
    def apply_gradients(self, gradients):
        self.__model.optimizer.apply_gradients(zip(gradients, self.__model.trainable_variables))
    

class ValueNetwork:
    def __init__(self, board_size=6, num_kernel=192, num_block=9,
                 model_path: str = None, optimizer=Adam(learning_rate=0.001)):
        if model_path is not None:
            # ファイルが指定されていれば，そこからモデルを読む．
            self.__model = load_model(model_path)
        else:
            # 引数で指定された盤面サイズとカーネル数に基づいてモデルを構築する.
            self.__BOARD_SIZE = board_size
            self.__NUM_KERNEL = num_kernel
            self.__NUM_BLOCK = num_block
            self.__model = self.__init_model()

        self.__model.compile(optimizer=optimizer)

    def call(self, x):
        return self.__model(x)

    def __del__(self):
        del self.__model

    def __init_model(self) -> Model:
        def conv(num_kernel=self.__NUM_KERNEL, kernel_size=3) -> Conv2D:
            return Conv2D(num_kernel, kernel_size, padding="same", use_bias=False,
                          kernel_initializer="he_normal", kernel_regularizer=L2(0.0005))

        def res_block():
            def f(x):
                sc = x
                x = conv()(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                x = conv()(x)
                x = BatchNormalization()(x)
                x = Add()([x, sc])
                x = Activation("relu")(x)
                return x
            return f

        input = Input(shape=(self.__BOARD_SIZE, self.__BOARD_SIZE, NN_NUM_CHANNEL))
        x = conv()(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        for _ in range(self.__NUM_BLOCK):
            x = res_block()(x)

        x = conv(1, 1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(256, kernel_initializer="he_normal", kernel_regularizer=L2(0.0005), activation="relu")(x)
        value = Dense(1, kernel_initializer="glorot_normal", kernel_regularizer=L2(0.0005), activation="tanh")(x)

        return Model(inputs=input, outputs=[value])
    
    @property
    def weights(self):
        return self.__model.trainable_variables
    
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
        ValueNetworkオブジェクトをパラメータごと複製したオブジェクトを返す(深いコピー).

        Returns
        -------
        copied: DualNetwork
        """
        return PolicyNetwork(src=self)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        ValueNetworkへの入力のバッチを受け取って，その出力を返す．
        """
        return self.__model.predict(x, verbose=0)

    def predict_from_position(self, pos: Position) -> np.ndarray:
        size = pos.SIZE
        x = np.zeros(shape=(1, size, size, NN_NUM_CHANNEL))
        position_to_input(pos, x[0])
        return self.predict(x, batch_size=1)
    
    def apply_gradients(self, gradients):
        self.__model.optimizer.apply_gradients(zip(gradients, self.__model.trainable_variables))
