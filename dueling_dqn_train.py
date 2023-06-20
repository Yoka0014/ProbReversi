"""
DQNの訓練スクリプト
"""
import math
import random

import numpy as np
import tensorflow as tf

from dueling_dqn import NN_NUM_CHANNEL, QNetwork, Episode, ReplayBuffer, position_to_input
from prob_reversi import Position, Move


class DQNConfig:
    """
    学習全般の設定
    """
    def __init__(self):
        self.board_size = 6     # 盤面サイズ
        self.trans_prob = [1.0] * self.board_size ** 2
        
        self.nn_optimizer = tf.optimizers.Adam(learning_rate=1.0e-5)    # QNetworkのオプティマイザ
        self.nn_loss_function = tf.losses.Huber()   # QNetworkの損失関数
        self.nn_num_kernel = 128
        self.nn_num_res_block = 2   # ResBlockの数

        self.batch_size = 256   # QNetworkに入力するバッチサイズ.
        self.target_net_update_interval = 5    # target networkを更新する間隔. (target_net_update_interval * batch_size)回のエピソード終了毎にtarget networkが更新される.
        self.train_steps = 100000     # NNのパラメータを何回更新するか.
        self.warmup_size = self.batch_size * 100    # ReplayBufferに何エピソード溜まったら学習を開始するか.
        self.replay_buffer_capacity = 30000     # Replay bufferのサイズ.

        self.discount_rate = 0.99   # 割引率
        self.epsilon_start = 0.9    # epsilonの初期値
        self.epsilon_end = 0.05     # epsilonの最小値
        self.epsilon_decay = 2000  # epsilonの減衰速度

        self.model_path = "qnet_{0}.h5"
        self.loss_path = "loss_history.txt"
        self.save_network_interval = 100    # Q-Networkのパラメータを保存する間隔. (checkpoint_interval * batch_size)回のエピソード終了毎に保存される.


class SharedStorage:
    """
    共有変数など

    Parameters
    ----------
    qnet: QNetwork
        学習対象のQNetwork

    target_net: QNetwork
        ターゲットネットワーク

    replay_buffer: ReplayBuffer
        経験再生バッファ

    episode_count: int
        今までのエピソード数

    train_count: int
        今までのQNetworkのパラメータ更新回数

    save_count: int
        NNの保存回数

    loss_histroy: list[float]
        損失の履歴
    """
    def __init__(self, config: DQNConfig):
        self.qnet = QNetwork(config.board_size, num_kernel=config.nn_num_kernel, num_res_block=config.nn_num_res_block, optimizer=config.nn_optimizer, loss=config.nn_loss_function)
        self.target_net = QNetwork(src=self.qnet)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity, config.board_size)

        self.episode_count = 0
        self.train_count = 0     # qnetのパラメータの更新回数.
        self.save_count = 0 
        self.loss_histroy: list[float] = []


def epsilon(config: DQNConfig, step_count: int):
    """
    epsilon-greedy方策で用いるepsilonを算出する.

    Parameters
    ----------
    config: DQNConfig
        学習全体の設定

    step_count: int
        今までのQNetworkの更新回数
        
    """
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-step_count / config.epsilon_decay)

def game_score_to_reward(score: int):
    if score == 0:
        return 0
    return 1 if score > 0 else -1


def exec_episodes(config: DQNConfig, shared: SharedStorage):
    """
    バッチサイズの分だけ，まとめて対局を行う.
    """
    board_size, batch_size = config.board_size, config.batch_size
    positions = [Position(config.board_size, trans_prob=config.trans_prob) for _ in range(batch_size)]
    batch = np.empty((batch_size, board_size, board_size , NN_NUM_CHANNEL)).astype(np.float32)
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
            if pass_count == 2: # パスが2連続で発生したら終局.
                continue

            terminated_all = False
            position_to_input(pos, x)

        if terminated_all:
            break

        q_batch = qnet.predict_q(batch)
        for i, (pos, q, episode) in enumerate(zip(positions, q_batch, episodes)):
            if pass_counts[i] == 2: # 終局している局面は無視.
                continue

            # epsilon-greedy方策に従って着手を決める.
            if random.random() < e:
                coord = pos.sample_next_move()  # random
            else:
                moves = list(pos.get_next_moves())
                coord = max(moves, key=lambda c: q[c]) if len(moves) != 0 else pos.PASS_COORD   # greedy

            if coord == pos.PASS_COORD:
                pass_counts[i] += 1
                pos.do_pass()
                episode.add_move(Move(coord=coord))

                if pass_counts[i] == 2:
                    episode.set_terminal_reward(pos.side_to_move, game_score_to_reward(pos.get_score()))
                    
                continue

            pass_counts[i] = 0
            move = pos.get_move(coord)
            pos.do_move(move)
            episode.add_move(move)

    for episode in episodes:
        replay_buffer.save_episode(episode)

    shared.episode_count += batch_size
                

def train(config: DQNConfig, shared: SharedStorage) -> bool:
    """
    shared.replay_bufferからconfig.batch_sizeのバッチを作り, それを用いてQNetworkのパラメータを1回更新する.
    """
    batch_size, board_size = config.batch_size, config.board_size
    qnet, target_net, replay_buffer = shared.qnet, shared.target_net, shared.replay_buffer

    batch = replay_buffer.sample_batch(batch_size)
    x = np.empty((batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32) 
    x_next = np.empty((batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32)
    td_targets = np.empty((batch_size, 1))
    terminal_rewards = np.empty((batch_size, 1))
    moves: list[Move] = []

    for i, (pos, _, next_pos, _, _) in enumerate(batch):
        position_to_input(pos, x[i])
        position_to_input(next_pos, x_next[i])
    
    actor_q = qnet.predict_q(x_next)
    target_q = target_net.predict_q(x_next)
    for i, (_, move, next_pos, terminal_reward, next_is_terminal) in enumerate(batch):
        aq = actor_q[i]
        tq = target_q[i]
        if not next_is_terminal:
            next_moves = list(next_pos.get_next_moves())
            if len(next_moves) != 0:
                td_targets[i] = tq[max(next_moves, key=lambda c: aq[c])]
            else:
                td_targets[i] = tq[next_pos.PASS_COORD]
            
            # 相手の手番から見た行動価値を計算しているので符号の反転が必要
            td_targets[i] = -config.discount_rate * td_targets[i]
        else:
            td_targets[i] = terminal_reward
        
        terminal_rewards[i] = terminal_reward
        moves.append(move)
        

    loss = qnet.train(x, moves, td_targets, terminal_rewards).numpy().item()
    shared.loss_histroy.append(loss)
    shared.train_count += 1

    print(f"loss: {loss}\n")


def main(config: DQNConfig):
    shared = SharedStorage(config)
    while shared.train_count < config.train_steps:
        # エピソードの実行
        exec_episodes(config, shared)

        if shared.train_count != 0 and shared.episode_count % config.target_net_update_interval == 0:
            # target netの更新
            shared.target_net = QNetwork(src=shared.qnet)
            print("Info: Target-Network has been updated.")

        if shared.train_count != 0 and shared.episode_count % config.save_network_interval == 0:
            # Q-Networkの保存
            path = config.model_path.format(shared.save_count)
            shared.qnet.save(path)
            shared.save_count += 1
            print(f"Info: Q-Network has been saved at \"{path}\"")

            with open(config.loss_path, "w") as file:
                file.write(f"{str(shared.loss_histroy)}\n")
            print(f"Info: History of losses has been saved at \"{config.loss_path}\"")

        if len(shared.replay_buffer) > config.warmup_size:  # 十分なエピソードが溜まっていないときは学習しない
            train(config, shared)
        
        tf.keras.backend.clear_session()    

    shared.qnet.save(config.model_path.format("final"))


if __name__ == "__main__":
    main(DQNConfig())
