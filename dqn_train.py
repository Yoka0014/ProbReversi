"""
DQNの訓練スクリプト
"""
import math

import numpy as np

from dqn import QNetwork, TDTarget, Episode, ReplayBuffer, position_to_input
from prob_reversi import Position, Move

class DQNConfig:
    def __init__(self):
        # self.board_size = 6
        # self.trans_prob = [0.8, 0.2, 0.5, 0.5, 0.2, 0.8,
        #                    0.2, 0.2, 0.5, 1.0, 0.2, 0.2,
        #                    0.5, 0.5, 1.0, 1.0, 1.0, 0.5,
        #                    0.5, 1.0, 1.0, 1.0, 0.5, 0.5,
        #                    0.8, 0.8, 1.0, 0.5, 0.8, 0.8,
        #                    0.2, 0.8, 0.5, 0.5, 0.8, 0.2]

        self.board_size = 4
        self.trans_prob = [1.0 for _ in range(16)]

        self.batch_size = 32
        self.train_steps = 100000     # NNを何ステップ分学習を行うか.
        self.window_size = 100000     # Replay bufferのサイズ.
        self.warmup_size = self.batch_size * 100    # ReplayBufferに何エピソード溜まったら学習を開始するか.
        self.checkpoint_interval = 128    # 何エピソード毎にtarget networkを更新するか.

        self.discount_rate = 0.99
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 200

        self.model_path = "qnet.h5"


class SharedValue:
    def __init__(self, config: DQNConfig):
        self.episode_count = 0
        self.qnet = QNetwork(config.board_size)
        self.target_net = self.qnet.copy()
        self.replay_buffer = ReplayBuffer(config.window_size, config.board_size)
        self.batch = (np.zeros((config.batch_size, 2, config.board_size, config.board_size)), 
                      np.zeros((config.batch_size, config.board_size ** 2)))


def epsilon(config: DQNConfig, step_count):
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-step_count / config.epsilon_decay)


def exec_episode(config: DQNConfig, shared: SharedValue):
    pos = Position(config.board_size, trans_prob=config.trans_prob)
    pass_count = 0
    x = np.empty((2, config.board_size, config.board_size)).astype("float32")
    
    while pass_count != 2:
        x.fill(0.0)
        position_to_input(pos, x)
        

def main(config: DQNConfig):
    pass
