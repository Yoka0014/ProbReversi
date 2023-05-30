"""
DQNの訓練スクリプト
"""
import math
import random

import numpy as np
import tensorflow as tf

from dqn import QNetwork, Episode, ReplayBuffer, position_to_input
from prob_reversi import Position, Move


class DQNConfig:
    def __init__(self):
        self.board_size = 6
        # self.trans_prob = [0.8, 0.2, 0.5, 0.5, 0.2, 0.8,
        #                    0.2, 0.2, 0.5, 1.0, 0.2, 0.2,
        #                    0.5, 0.5, 1.0, 1.0, 1.0, 0.5,
        #                    0.5, 1.0, 1.0, 1.0, 0.5, 0.5,
        #                    0.8, 0.8, 1.0, 0.5, 0.8, 0.8,
        #                    0.2, 0.8, 0.5, 0.5, 0.8, 0.2]
        self.trans_prob = [1.0 for _ in range(36)]

        self.batch_size = 32
        self.step_size = 16     # 何エピソードごとに学習を行うか.
        self.train_steps = 10000     # NNを何ステップ分学習を行うか.
        self.window_size = 2000     # Replay bufferのサイズ.
        self.warmup_size = self.batch_size * 10    # ReplayBufferに何エピソード溜まったら学習を開始するか.
        self.checkpoint_interval = 100    # 何エピソード毎にtarget networkを更新するか.

        self.discount_rate = 0.99   # 割引率
        self.epsilon_start = 0.9    # epsilonの初期値
        self.epsilon_end = 0.05     # epsilonの最小値
        self.epsilon_decay = 2000  # epsilonの減衰速度

        self.num_evaluation_game = 400  # 強さを評価する際に行う対局数

        self.model_path = "qnet_{0}.h5"


class SharedValue:
    def __init__(self, config: DQNConfig):
        self.episode_count = 0
        self.qnet = QNetwork(config.board_size)
        self.qnet.save("qnet_init.h5")
        self.target_net = QNetwork(model_path="qnet_init.h5")
        self.replay_buffer = ReplayBuffer(config.window_size, config.board_size)
        self.step_count = 0
        self.save_count = 0 


def epsilon(config: DQNConfig, step_count):
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-step_count / config.epsilon_decay)


def exec_episode(config: DQNConfig, shared: SharedValue):
    print(f"episode = {shared.episode_count + 1}")

    pos = Position(config.board_size, trans_prob=config.trans_prob)
    x = np.empty((1, config.board_size, config.board_size, 2)).astype("float32")

    episode = Episode(pos)
    replay_buffer = shared.replay_buffer
    qnet = shared.qnet
    e = epsilon(config, shared.step_count)

    print(f"epsilon = {e}")

    pass_count = 0
    while pass_count != 2:
        # epsilon-greedy
        if random.random() < e:
            coord = pos.sample_next_move()
        else:
            position_to_input(pos, x[0])
            q = qnet.predict(x, batch_size=1)[0]
            moves = list(pos.get_next_moves())
            coord = max(moves, key=lambda c: q[c]) if len(moves) != 0 else pos.PASS_COORD

        if coord == pos.PASS_COORD:
            pass_count += 1
            episode.add_move(Move(coord=pos.PASS_COORD))
            continue

        pass_count = 0
        move = pos.get_move(coord)
        pos.do_move(move)
        episode.add_move(move)
        
    shared.episode_count += 1
    replay_buffer.save_episode(episode)


def train(config: DQNConfig, shared: SharedValue) -> bool:
    batch_size, board_size = config.batch_size, config.board_size
    qnet, target_net, replay_buffer = shared.qnet, shared.target_net, shared.replay_buffer
    if len(replay_buffer) < config.warmup_size:  # 十分なエピソードが溜まっていない.
        return False

    batch = replay_buffer.sample_batch(batch_size)
    q_x = np.empty((batch_size, board_size, board_size, 2)).astype("float32")
    target_x = np.empty((batch_size, board_size, board_size, 2)).astype("float32")

    for i, (pos, _, next_pos, _) in enumerate(batch):
        position_to_input(pos, q_x[i])
        position_to_input(next_pos, target_x[i])
        
    target_out = target_net.predict(target_x, batch_size)
    td_targets = np.zeros((batch_size, board_size ** 2 + 1))
    for i, (_, move, next_pos, reward) in enumerate(batch):
        if reward is None:
            moves = list(next_pos.get_next_moves())
            if len(moves) != 0:
                td_targets[i][move.coord] = target_out[i][max(moves, key=lambda c: target_out[i][c])]
            else:
                td_targets[i][move.coord] = target_out[i][next_pos.PASS_COORD]
            
            # 相手の手番から見た行動価値を計算しているので符号の反転が必要
            td_targets[i][move.coord] = -config.discount_rate * td_targets[i][move.coord]
        else:
            td_targets[i][move.coord] = -reward

    qnet.train(q_x, td_targets, list(map(lambda b: b[1], batch)))
    shared.step_count += 1

def main(config: DQNConfig):
    shared = SharedValue(config)
    while shared.step_count < config.train_steps:
        exec_episode(config, shared)

        if shared.step_count != 0 and shared.episode_count % config.checkpoint_interval == 0:
            # target netの更新
            path = config.model_path.format(shared.save_count)
            shared.qnet.save(path)
            shared.save_count += 1
            shared.target_net = QNetwork(model_path=path)

        if shared.episode_count % config.step_size == 0:
            train(config, shared)

    shared.qnet.save(config.model_path.format("final"))


if __name__ == "__main__":
    main(DQNConfig())
