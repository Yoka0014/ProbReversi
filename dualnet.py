# ====================
# デュアルネットワークの作成
# ====================

# パッケージのインポート
import numpy as np

from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
import os

from prob_reversi import Position

# パラメータの準備
BOARD_SIZE = 6  # 盤面のサイズ
DN_FILTERS = 128  # 畳み込み層のカーネル数（本家は256）
DN_RESIDUAL_NUM = 9  # 残差ブロックの数（本家は19）
DN_NUM_CHANNEL = 2
DN_INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, DN_NUM_CHANNEL)  # 入力シェイプ
DN_OUTPUT_SIZE = BOARD_SIZE**2 + 1  # 行動数(配置先(n*n)+パス(1))


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
        dest = np.empty(shape=(pos.SIZE, pos.SIZE, DN_NUM_CHANNEL))
    elif dest.shape != (pos.SIZE, pos.SIZE, DN_NUM_CHANNEL):
        raise ValueError("The shape of dest must be (pos.SIZE, pos.SIZE, 2).")

    dest.fill(0.0)
    for coord in pos.get_player_disc_coords():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][0] = 1.0

    for coord in pos.get_opponent_disc_coords():
        x, y = pos.convert_coord1D_to_coord2D(coord)
        dest[x][y][1] = 1.0

    return dest


# 畳み込み層の作成
def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False,
                  kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

# 残差ブロックの作成


def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f

# デュアルネットワークの作成


def dual_network():
    # 入力層
    input = Input(shape=DN_INPUT_SHAPE)

    # 畳み込み層
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 残差ブロック x 16
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    # プーリング層
    x = GlobalAveragePooling2D()(x)

    # ポリシー出力
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='softmax', name='pi')(x)

    # バリュー出力
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    # モデルの作成
    return Model(inputs=input, outputs=[p, v])
