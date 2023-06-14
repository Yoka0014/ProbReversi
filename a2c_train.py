"""
A2Cの訓練スクリプト
"""
import random

import numpy as np
import tensorflow as tf

from prob_reversi import Position, DiscColor, Move
from dualnet import NN_NUM_CHANNEL, DualNetwork, position_to_input


class A2CConfig:
    """
    学習全般の設定
    """

    def __init__(self):
        self.board_size = 6     # 盤面サイズ
        self.trans_prob = [1.0] * self.board_size ** 2

        self.nn_optimizer = tf.optimizers.Adam(learning_rate=0.001)    # DualNetworkのオプティマイザ

        self.batch_size = 32   # DualNetworkに入力するバッチサイズ.
        self.train_steps = 10000     # NNのパラメータを何回更新するか.

        self.model_path = "dualnet_{0}.h5"
        self.save_network_interval = 100    # DualNetworkのパラメータを保存する間隔. save_network_interval回のNNの更新が行われたら保存する.


class SharedStorage:
    def __init__(self, config: A2CConfig):
        self.dual_net = DualNetwork(config.board_size, optimizer=config.nn_optimizer)

        self.episode_count = 0
        self.train_count = 0     # dual_netのパラメータの更新回数.
        self.save_count = 0


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
        self.__reward_from_black = 0.0

    @property
    def reward_from_black(self) -> float:
        return self.__reward_from_black

    def len(self):
        return len(self.__history)

    def add_move(self, move: Move):
        self.__history.append(move)

    def set_reward(self, player: DiscColor, score: int):
        if score == 0:
            self.__reward_from_black = 0.0
        self.__reward_from_black = 1.0 if score > 0 else -1.0

        if player != DiscColor.BLACK:
            self.__reward_from_black *= -1

    def sample_state(self) -> tuple[Position, Move, float]:
        """
        エピソードの中から，状態(局面)を1つだけサンプリングし, (局面，着手, 終端報酬)のペアを返す.
        """
        i = random.randint(0, len(self.__history) - 1)
        pos = self.__root_pos.copy()
        for move in self.__history[0:i]:
            if move.coord == pos.PASS_COORD:
                pos.do_pass()
            else:
                pos.do_move(move)

        reward = self.__reward_from_black
        if pos.side_to_move != DiscColor.BLACK:
            reward *= -1

        return pos, self.__history[i], reward


def exec_episodes(config: A2CConfig, shared: SharedStorage) -> list[tuple[Position, Move, float]]:
    board_size, batch_size = config.board_size, config.batch_size
    positions = [Position(config.board_size, trans_prob=config.trans_prob) for _ in range(batch_size)]
    batch = np.empty((batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32)
    episodes = list(map(lambda pos: Episode(pos), positions))
    dual_net = shared.dual_net

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

        pv_batch = dual_net.predict(batch, batch_size=batch_size)
        for i, (pos, policy, episode) in enumerate(zip(positions, pv_batch[0], episodes)):
            if pass_counts[i] == 2:  # 終局している局面は無視.
                episode.set_reward(pos.side_to_move, pos.get_score())
                continue

            # 方策の確率分布に従って着手を決める.
            moves = list(pos.get_next_moves())
            if len(moves) != 0:
                prob = policy[moves]
                prob_sum = np.sum(prob)
                if prob_sum != 0:
                    np.divide(prob, prob_sum, out=prob)
                else:
                    prob.fill(1.0 / len(moves))
                coord = np.random.choice(moves, p=prob).item()
                pass_counts[i] = 0
                move = pos.get_move(coord)
                pos.do_move(move)
                episode.add_move(move)
            else:
                pass_counts[i] += 1
                pos.do_pass()
                episode.add_move(Move(coord=pos.PASS_COORD))

    shared.episode_count += batch_size

    return list(map(lambda e: e.sample_state(), episodes))


def train(config: A2CConfig, shared: SharedStorage, train_data: list[tuple[Position, Move, float]]):
    batch_size, board_size = config.batch_size, config.board_size
    dual_net = shared.dual_net

    batch = np.empty(shape=(batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32)
    move_coords = list(map(lambda d: d[1].coord, train_data))
    rewards = np.empty(shape=(batch_size, 1)).astype(np.float32)

    for i, (pos, _, reward) in enumerate(train_data):
        batch[i] = position_to_input(pos)
        rewards[i][0] = reward

    dual_net.train_with_experience(batch, move_coords, rewards)
    shared.train_count += 1

def main(config: A2CConfig, model_path: str = None):
    shared = SharedStorage(config)
    
    if model_path is not None:
        shared.dual_net = DualNetwork(model_path=model_path)

    while shared.train_count < config.train_steps:
        # エピソードの実行
        train_data = exec_episodes(config, shared)

        if shared.train_count != 0 and shared.train_count % config.save_network_interval == 0:
            path = config.model_path.format(shared.save_count)
            shared.dual_net.save(path)
            shared.save_count += 1
            print(f"Info: Q-Network has been saved at \"{path}\"")
        
        train(config, shared, train_data)
        tf.keras.backend.clear_session()

    shared.dual_net.save(config.model_path.format("final"))


if __name__ == "__main__":
    main(A2CConfig())