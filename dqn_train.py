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

        self.batch_size = 256
        self.checkpoint_interval = 5    # target networkを更新する間隔. (checkpoint_interval * batch_size)回のエピソード終了毎にtarget networkが更新される.
        self.train_steps = 10000     # NNのパラメータを何回更新するか.
        self.warmup_size = self.batch_size * 100    # ReplayBufferに何エピソード溜まったら学習を開始するか.
        self.replay_buffer_capasity = 1000000 // (self.board_size ** 2 - 4)     # Replay bufferのサイズ.

        self.discount_rate = 0.99   # 割引率
        self.epsilon_start = 0.9    # epsilonの初期値
        self.epsilon_end = 0.05     # epsilonの最小値
        self.epsilon_decay = 2000  # epsilonの減衰速度

        self.num_evaluation_game_batch = 2  # 強さを評価する際に行う対局のバッチ数. num_evaluation_game_batch * batch_size 分の対局が行われる.

        self.model_path = "qnet_{0}.h5"


class SharedStorage:
    def __init__(self, config: DQNConfig):
        self.qnet = QNetwork(config.board_size)
        self.target_net = QNetwork(src=self.qnet)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capasity, config.board_size)

        self.episode_count = 0
        self.train_count = 0     # qnetのパラメータの更新回数.
        self.save_count = 0 

        self.best_model_path = "NULL"
        self.best_model_winrate = -float("inf")


def epsilon(config: DQNConfig, step_count):
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-step_count / config.epsilon_decay)


def exec_episodes(config: DQNConfig, shared: SharedStorage):
    """
    バッチサイズの分だけ，まとめて対局を行う.
    """
    board_size, batch_size = config.board_size, config.batch_size
    positions = [Position(config.board_size, trans_prob=config.trans_prob) for _ in range(batch_size)]
    batch = np.empty((batch_size, board_size, board_size , 2)).astype(np.float32)
    episodes = list(map(lambda pos: Episode(pos), positions))
    replay_buffer = shared.replay_buffer
    qnet = shared.qnet
    e = epsilon(config, shared.train_count)

    episode_id = shared.episode_count + 1
    print(f"episodes: {episode_id} to {episode_id + batch_size - 1}")
    print(f"epsilon = {e}")

    pass_counts = [0] * batch_size
    while True:
        terminated_all = True
        for pos, x, pass_count in zip(positions, batch, pass_counts):
            if pass_count == 2:
                continue

            terminated_all = False
            position_to_input(pos, x)

        if terminated_all:
            break

        q_batch = qnet.predict(batch, batch_size=batch_size)
        for i, (pos, q, episode) in enumerate(zip(positions, q_batch, episodes)):
            if pass_counts[i] == 2:
                continue

            # epsilon-greedy
            if random.random() < e:
                coord = pos.sample_next_move()
            else:
                moves = list(pos.get_next_moves())
                coord = max(moves, key=lambda c: q[c]) if len(moves) != 0 else pos.PASS_COORD

            if coord == pos.PASS_COORD:
                pass_counts[i] += 1
                pos.do_pass()
                episode.add_move(Move(coord=coord))
                continue

            pass_counts[i] = 0
            move = pos.get_move(coord)
            pos.do_move(move)
            episode.add_move(move)

    for episode in episodes:
        replay_buffer.save_episode(episode)

    shared.episode_count += batch_size
                

def train(config: DQNConfig, shared: SharedStorage) -> bool:
    batch_size, board_size = config.batch_size, config.board_size
    qnet, target_net, replay_buffer = shared.qnet, shared.target_net, shared.replay_buffer

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
    shared.train_count += 1


def main(config: DQNConfig):
    shared = SharedStorage(config)
    while shared.train_count < config.train_steps:
        exec_episodes(config, shared)

        if shared.train_count != 0 and shared.episode_count % config.checkpoint_interval == 0:
            # target netの更新
            shared.qnet.save(config.model_path.format(shared.save_count))
            shared.save_count += 1
            shared.target_net = QNetwork(src=shared.qnet)

        
        if len(shared.replay_buffer) > config.warmup_size:  # 十分なエピソードが溜まっていない.
            train(config, shared)

    shared.qnet.save(config.model_path.format("final"))


if __name__ == "__main__":
    main(DQNConfig())
