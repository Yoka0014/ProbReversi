import random
import pickle

import numpy as np

import tensorflow as tf

from prob_reversi import Position, Move, DiscColor
from dqn import QNetwork, Episode, position_to_input, NN_NUM_CHANNEL


class TrainDataGeneratorConfig:
    def __init__(self):
        self.board_size = 6
        self.num_squares = self.board_size ** 2

        t = 5
        self.trans_prob = []
        for coord in range(self.board_size ** 2):
            self.trans_prob.append(1.0 - t * 0.01 * (coord % (self.board_size + 1) + 3))

        self.model_path = "qnet_6x6_best.h5"
        self.out_path = "train_data_6x6.pickle" 

        self.batch_size = 4096
        self.num_batches = 5000

        self.softmax_temperature = 0.01
        self.num_prob_moves = (self.board_size ** 2 - 4) // 6


class SharedStorage:
    def __init__(self) -> None:
        self.qnet: QNetwork = None
        self.episode_count = 0


def softmax(x, axis=0, t=1.0):
    exp_x = np.exp((x - np.max(x, axis=axis)) / t)
    exp_sum = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / exp_sum


def score_to_reward(score: int) -> float:
    if score == 0:
        return 0.0
    return 1.0 if score > 0 else -1.0


def generate_batch(config: TrainDataGeneratorConfig, shared: SharedStorage):
    board_size, batch_size = config.board_size, config.batch_size
    positions = [Position(config.board_size, trans_prob=config.trans_prob) for _ in range(batch_size)]
    batch = np.empty((batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32)
    episodes = list(map(lambda pos: Episode(pos), positions))
    rewards = [0.0] * batch_size
    qnet = shared.qnet

    episode_id = shared.episode_count + 1
    print(f"episodes: {episode_id} to {episode_id + batch_size - 1}")

    pass_counts = [0] * batch_size
    while True:
        terminated_all = True
        for pos, x, pass_count in zip(positions, batch, pass_counts):
            if pass_count == 2: # パスが2連続で発生したら終局.
                continue

            terminated_all = False
            position_to_input(pos, x)

        if terminated_all:
            break

        q_batch = qnet.predict(batch, batch_size=batch_size)
        for i, (pos, q, episode) in enumerate(zip(positions, q_batch, episodes)):
            if pass_counts[i] == 2: # 終局している局面は無視.
                continue

            moves = list(pos.get_next_moves())

            if len(moves) != 0:
                move_num = pos.empty_square_count - pos.SQUARE_NUM + 4
                if move_num < config.num_prob_moves:
                    policy = softmax(q[moves], t=config.softmax_temperature).astype(np.float64)
                    np.divide(policy, np.sum(policy), out=policy)
                    coord = np.random.choice(moves, p=policy).item() if len(moves) != 0 else pos.PASS_COORD
                else:
                    coord = max(moves, key=lambda m: q[m])
            else:
                coord = pos.PASS_COORD

            if coord == pos.PASS_COORD:
                pass_counts[i] += 1
                if(pass_counts[i] == 2):
                    rewards[i] = score_to_reward(pos.get_score_from(DiscColor.BLACK))
                pos.do_pass()
                episode.add_move(Move(coord=coord))
                continue

            pass_counts[i] = 0
            move = pos.get_move(coord)
            pos.do_move(move)
            episode.add_move(move)

    shared.episode_count += batch_size
    train_batch = [e.sample_state() for e in episodes]
    train_batch = [(p.bitboard, m.coord, r if p.side_to_move == DiscColor.BLACK else -r) for (p, m), r in zip(train_batch, rewards)]
    return train_batch

def main():
    config = TrainDataGeneratorConfig()
    shared = SharedStorage()
    shared.qnet = QNetwork(model_path=config.model_path)

    with open(config.out_path, "wb") as file:
        for _ in range(config.num_batches):
            batch = generate_batch(config, shared)
            for item in batch:
                pickle.dump(item, file)
            file.flush()
            tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
