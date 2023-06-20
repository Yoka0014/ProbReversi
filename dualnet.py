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


class DualNetwork:
    """
    方策と価値を出力するデュアルネットワーク
    """

    def __init__(self, board_size=6, kernel_num=192, num_block=9,
                 model_path: str = None, src=None, copy_optimizer=False, optimizer=Adam(learning_rate=0.001), policy_entropy_factor=0.05):
        """
        コンストラクタ

        Parameters
        ----------
        board_size: int
            盤面のサイズ(デフォルトは6)

        kernel_num: int
            カーネル(フィルター)数(デフォルトは192)

        num_block: int
            残差ブロックの数(デフォルトは9)

        model_path: str
            ファイルからパラメータをロードする場合のファイルパス(Noneなら無視される)

        src: DualNetwork
            別のDualNetworkオブジェクトからパラメータをコピーする際のコピー元(Noneなら無視される)

        copy_optimizer: bool
            srcからオプティマイザもコピーするかどうか

        optimizer: tf.Optimizer
            学習の際に用いるオプティマイザ(デフォルトはAdam)

        policy_entropy_factor: float
            方策のエントロピーが小さくならないようにするためのペナルティーの強さ
        """
        if model_path is not None:
            # ファイルが指定されていれば，そこからモデルを読む．
            self.__model = load_model(model_path)
        elif src is not None and type(src) is DualNetwork:
            # 引数でコピー元のQNetworkが指定されていれば，そのパラメータをself.__modelにコピーする．
            self.__BOARD_SIZE = src.__model.input_shape[0]
            self.__POLICY_ENTROPY_FACTOR = policy_entropy_factor
            self.__model = clone_model(src.__model)
            self.__model.set_weights(src.__model.get_weights())
            if copy_optimizer:
                optimizer = src.__model.optimizer
        else:
            # 引数で指定された盤面サイズとカーネル数に基づいてモデルを構築する.
            self.__BOARD_SIZE = board_size
            self.__NUM_KERNEL = kernel_num
            self.__NUM_BLOCK = num_block
            self.__POLICY_ENTROPY_FACTOR = policy_entropy_factor
            self.__model = self.__init_model()

        self.__model.compile(optimizer=optimizer)

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

        policy = conv(2, 1)(x)
        policy = BatchNormalization()(policy)
        policy = Activation("relu")(policy)
        policy = Flatten()(policy)
        # softmaxではなく, 直接ロジットを出力する.
        policy = Dense(self.__BOARD_SIZE ** 2 + 1, kernel_initializer="he_normal", kernel_regularizer=L2(0.0005))(policy)

        value = conv(1, 1)(x)
        value = BatchNormalization()(value)
        value = Activation("relu")(value)
        value = Flatten()(value)
        value = Dense(256, kernel_initializer="he_normal", kernel_regularizer=L2(0.0005), activation="relu")(value)
        value = Dense(1, kernel_initializer="glorot_normal", kernel_regularizer=L2(0.0005), activation="tanh")(value)

        return Model(inputs=input, outputs=[policy, value])

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
        DualNetworkオブジェクトをパラメータごと複製したオブジェクトを返す(深いコピー).

        Returns
        -------
        copied: DualNetwork
        """
        return DualNetwork(src=self)

    def predict(self, x: np.ndarray, batch_size=32) -> np.ndarray:
        """
        DualNetworkへの入力のバッチを受け取って，その出力を返す．
        """
        return self.__model.predict(x, batch_size=batch_size, verbose=0)

    def predict_from_position(self, pos: Position) -> np.ndarray:
        size = pos.SIZE
        x = np.zeros(shape=(1, size, size, NN_NUM_CHANNEL))
        position_to_input(pos, x[0])
        return self.predict(x, batch_size=1)

    def train_with_data(self, x: np.ndarray, policy_targets: np.ndarray, value_targets: np.ndarray):
        """
        与えられた教師データバッチから勾配計算を1回行って重みを更新する.
        """
        model = self.__model
        with tf.GradientTape() as tape:
            p, v = model(x)
            loss = CategoricalCrossentropy()(policy_targets, p) + MSE(value_targets, v)

        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    def train_with_experience(self, x: np.ndarray, move_coords: list[int], td_targets: np.ndarray, legal_moves: list[list[int]]):
        """
        実際に行った着手と価値のターゲットのバッチから勾配計算を1回行って重みを更新する.
        """
        EPSILON = 1.0e-7   # log(0)回避用の定数

        model = self.__model
        with tf.GradientTape() as tape:
            p, v = model(x)
            advantages = td_targets - v
            v_loss = tf.losses.Huber()(td_targets, v)

            p_loss_list = []
            for logits, moves, move_coord in zip(p, legal_moves, move_coords):
                probs = tf.gather(params=logits, indices=moves)
                probs = tf.nn.softmax(probs)
                p_loss_list.append(tf.math.log(probs[moves.index(move_coord)] + EPSILON))

            p_loss = tf.Variable(p_loss_list)
            p_loss = -tf.expand_dims(p_loss, axis=1) * tf.stop_gradient(advantages)
            p = tf.nn.softmax(p)
            entropy = -tf.reduce_sum(p * tf.math.log(p + EPSILON), axis=1, keepdims=True)
            loss = tf.reduce_mean(p_loss - self.__POLICY_ENTROPY_FACTOR * entropy) + v_loss

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"total_loss: {loss.numpy().item()}")
        return p_loss, v_loss