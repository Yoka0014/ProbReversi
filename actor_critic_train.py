"""
ActorCriticの訓練スクリプト
"""
import random

import numpy as np
import tensorflow as tf

from prob_reversi import Position, DiscColor, Move
from policy_value_net import NN_NUM_CHANNEL, PolicyNetwork, ValueNetwork, position_to_input


class ActorCriticConfig:
    """
    学習全般の設定
    """

    def __init__(self):
        self.board_size = 6     # 盤面サイズ

        self.trans_prob = [1.0] * self.board_size ** 2
        # t = 5
        # self.trans_prob = []
        # for coord in range(self.board_size ** 2):
        #     self.trans_prob.append(1.0 - t * 0.01 * (coord % (self.board_size + 1) + 3))

        self.nn_num_res_block = 3
        self.nn_optimizer = tf.optimizers.Adam(learning_rate=0.001)    # DualNetworkのオプティマイザ
        self.policy_entropy_factor = 0.0  # 方策のエントロピーが小さくなった際に与えるペナルティーの強さ
        self.discount_rate = 0.99   # 割引率
        self.use_reward_as_value_target = True  # ValueNetworkのターゲットにTDターゲットではなく, 報酬を用いるかどうか

        self.batch_size = 32   # DualNetworkに入力するバッチサイズ.
        self.train_steps = 10000     # NNのパラメータを何回更新するか.

        self.policy_model_path = "policy_net_{0}.h5"
        self.value_model_path = "value_net_{0}.h5"
        self.loss_path = "loss_history.txt"
        self.save_network_interval = 100    # DualNetworkのパラメータを保存する間隔. save_network_interval回のNNの更新が行われたら保存する.


class SharedStorage:
    def __init__(self, config: ActorCriticConfig):
        self.policy_net = PolicyNetwork(config.board_size, num_block=config.nn_num_res_block, optimizer=config.nn_optimizer)
        self.value_net = ValueNetwork(config.board_size, num_block=config.nn_num_res_block, optimizer=config.nn_optimizer)

        self.episode_count = 0
        self.train_count = 0     
        self.save_count = 0
        
        self.loss_history: tuple[list[float], list[float]] = ([], [])


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
        self.__history: list[tuple[Move, float]] = []  # 着手と状態価値の履歴.
        self.__reward_from_black = 0.0

    @property
    def reward_from_black(self) -> float:
        return self.__reward_from_black

    def len(self):
        return len(self.__history)

    def add_move(self, move: Move, value: float):
        self.__history.append((move, value))

    def set_reward(self, player: DiscColor, score: int):
        if score == 0:
            self.__reward_from_black = 0.0
        self.__reward_from_black = 1.0 if score > 0 else -1.0

        if player != DiscColor.BLACK:
            self.__reward_from_black *= -1

    def sample_state(self) -> tuple[Position, Move, float, float]:
        """
        エピソードの中から，状態(局面)を1つだけサンプリングし, (局面，着手, 状態価値, 終端報酬)のペアを返す.
        """
        i = random.randint(0, len(self.__history) - 1)
        pos = self.__root_pos.copy()
        for move, _ in self.__history[0:i]:
            if move.coord == pos.PASS_COORD:
                pos.do_pass()
            else:
                pos.do_move(move)

        reward = self.__reward_from_black
        if pos.side_to_move != DiscColor.BLACK:
            reward *= -1

        move = self.__history[i][0]
        tail = len(self.__history) - 1
        if tail - 2 <= i <= tail:
            value = self.__reward_from_black
            if pos.side_to_move == DiscColor.WHITE:
                value *= -1
        else:
            value = -self.__history[i + 1][1]

        reward = self.__reward_from_black
        if pos.side_to_move == DiscColor.WHITE:
            reward *= -1

        return pos, move, value, reward
    

def exec_episodes(config: ActorCriticConfig, shared: SharedStorage) -> list[tuple[Position, Move, float, float]]:
    board_size, batch_size = config.board_size, config.batch_size
    positions = [Position(config.board_size, trans_prob=config.trans_prob) for _ in range(batch_size)]
    batch = np.empty((batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32)
    episodes = list(map(lambda pos: Episode(pos), positions))
    policy_net, value_net = shared.policy_net, shared.value_net

    episode_id = shared.episode_count + 1
    print(f"episodes: {episode_id} to {episode_id + batch_size - 1}")

    pass_counts = [0] * batch_size
    while True:
        terminated_all = True
        for pos, x, pass_count in zip(positions, batch, pass_counts):
            if pass_count == 2:  # パスが2連続で発生したら終局.
                continue

            terminated_all = False
            position_to_input(pos, x)

        if terminated_all:
            break

        logits = policy_net.predict_logit(batch)
        values = value_net.predict(batch)
        for i, (pos, p_logit, value, episode) in enumerate(zip(positions, logits, values, episodes)):
            if pass_counts[i] == 2:  # 終局している局面は無視.
                episode.set_reward(pos.side_to_move, pos.get_score())
                continue

            # 方策の確率分布に従って着手を決める.
            moves = list(pos.get_next_moves())
            if len(moves) != 0:
                policy = tf.nn.softmax(p_logit[moves]).numpy().astype(np.float64)
                # probを64bitにした後に再度, 和が1になるように正規化しないとnp.random.choiceでエラーが出る.
                coord = np.random.choice(moves, p=policy / np.sum(policy)).item()
                pass_counts[i] = 0
                move = pos.get_move(coord)
                pos.do_move(move)
                episode.add_move(move, value[0].item())
            else:
                pass_counts[i] += 1
                pos.do_pass()
                episode.add_move(Move(coord=pos.PASS_COORD), value[0].item())

    shared.episode_count += batch_size

    return list(map(lambda e: e.sample_state(), episodes))


def train(config: ActorCriticConfig, shared: SharedStorage, train_data: list[tuple[Position, Move, float, float]]):
    EPSILON = 1.0e-7

    batch_size, board_size = config.batch_size, config.board_size
    policy_net, value_net = shared.policy_net, shared.value_net

    batch = np.empty(shape=(batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32)
    move_coords = list(map(lambda d: d[1].coord, train_data))
    td_targets = np.empty(shape=(batch_size, 1)).astype(np.float32)

    for i, (pos, _, value, reward) in enumerate(train_data):
        position_to_input(pos, batch[i])
        td_targets[i][0] = reward if config.use_reward_as_value_target else config.discount_rate * value

    masks = tf.one_hot(move_coords, board_size ** 2 + 1)
    with tf.GradientTape() as tape:
        p_logits = policy_net.call(batch)
        values = value_net.call(batch)
        advantages = td_targets - values

        v_loss = tf.reduce_mean(advantages ** 2)

        policies = tf.nn.softmax(p_logits)
        #p_loss = -masks * tf.math.log(policies + EPSILON) * tf.stop_gradient(advantages)
        p_loss = -masks * tf.math.log(policies + EPSILON) * tf.stop_gradient(td_targets)
        entropy = -tf.reduce_sum(policies * tf.math.log(policies + EPSILON), axis=1, keepdims=True)
        p_loss -= config.policy_entropy_factor * entropy
        p_loss = tf.reduce_mean(p_loss)

    p_grads, v_grads = tape.gradient([p_loss, v_loss], [policy_net.weights, value_net.weights])

    policy_net.apply_gradients(p_grads)
    value_net.apply_gradients(v_grads)

    print(f"policy_loss: {p_loss.numpy().item()}")
    print(f"value_loss: {v_loss.numpy().item()}\n")

    shared.loss_history[0].append(p_loss)
    shared.loss_history[1].append(v_loss)
    shared.train_count += 1

def main(config: ActorCriticConfig, policy_model_path: str = None, value_model_path: str = None):
    shared = SharedStorage(config)
    
    if policy_model_path is not None:
        shared.value_net = PolicyNetwork(model_path=policy_model_path)

    if value_model_path is not None:
        shared.value_net = ValueNetwork(model_path=value_model_path)

    while shared.train_count < config.train_steps:
        # エピソードの実行
        train_data = exec_episodes(config, shared)

        if shared.train_count != 0 and shared.train_count % config.save_network_interval == 0:
            path = config.policy_model_path.format(shared.save_count)
            shared.policy_net.save(path)
            print(f"Info: PolicyNetwork has been saved at \"{path}\"")

            path = config.value_model_path.format(shared.save_count)
            shared.value_net.save(path)
            print(f"Info: ValueNetwork has been saved at \"{path}\"")

            shared.save_count += 1

            with open(config.loss_path, 'w') as file:
                file.write(str(shared.loss_history))
            print(f"Info: loss histroy has been saved at \"{config.loss_path}\"")
        
        train(config, shared, train_data)
        tf.keras.backend.clear_session()

    shared.policy_net.save(config.policy_model_path.format("final"))
    shared.value_net.save(config.value_model_path.format("final"))


if __name__ == "__main__":
    main(ActorCriticConfig())