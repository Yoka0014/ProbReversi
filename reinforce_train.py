"""
REINFORCEの訓練スクリプト
"""
import random
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np
import tensorflow as tf

from prob_reversi import Position, DiscColor, Move
from policy_value_net import NN_NUM_CHANNEL, PolicyNetwork, position_to_input


class ReinforceConfig:
    def __init__(self):
        self.board_size = 6     # 盤面サイズ

        self.trans_prob = [1.0] * self.board_size ** 2
        # t = 5
        # self.trans_prob = []
        # for coord in range(self.board_size ** 2):
        #     self.trans_prob.append(1.0 - t * 0.01 * (coord % (self.board_size + 1) + 3))

        self.nn_num_res_block = 3
        self.nn_optimizer = tf.optimizers.Adam(learning_rate=0.001)    # NNのオプティマイザ
        self.ppo_clip_range = 0.1
        self.policy_entropy_factor = 0.005  # 方策のエントロピーが小さくなった際に与えるペナルティーの強さ
        self.policy_pool_size = 10  # 直近のPolicyNetworkをいくつまで残すか

        self.num_episode_in_batch = 256   # バッチあたりのエピソード数
        self.train_steps = 10000     # NNのパラメータを何回更新するか

        self.policy_model_path = "policy_net_{0}.h5"
        self.loss_path = "reward_history.txt"
        self.checkpoint_interval = 100    # NNのパラメータを保存する間隔. save_network_interval回のNNの更新が行われたら保存し, PolicyPoolに追加する.


class Policy(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def set_board_size(self, board_size: int):
        pass

    @abstractmethod
    def set_batch_size(self, batch_size: int):
        pass

    @abstractmethod
    def sample_moves(self, positions: list[Position]) -> list[int]:
        pass


class PolicyPool:
    def __init__(self, capasity: int):
        self.__deque: deque[Policy] = deque(maxlen=capasity)

    def add(self, policy: Policy):
        self.__deque.append(policy)

    def sample(self) -> Policy:
        return random.choice(self.__deque)


class SharedStorage:
    def __init__(self, config: ReinforceConfig):
        self.policy_pool = PolicyPool(config.policy_pool_size)
        self.policy_pool.add(RandomPolicy())
        self.policy_net = PolicyNetwork(
            config.board_size, num_block=config.nn_num_res_block, optimizer=config.nn_optimizer)

        self.episode_count = 0
        self.train_count = 0
        self.save_count = 0
        self.reward_histroy = []


class RandomPolicy(Policy):
    def set_board_size(self, board_size: int):
        pass

    def set_batch_size(self, batch_size: int):
        pass

    def sample_moves(self, positions: list[Position]) -> list[int]:
        moves = []
        for pos in positions:
            next_moves = list(pos.get_next_moves())
            if len(next_moves) == 0:
                moves.append(pos.PASS_COORD)
            else:
                moves.append(random.choice(next_moves))
        return moves


class NetworkPolicy(Policy):
    def __init__(self, policy_net: PolicyNetwork):
        self.__net = policy_net
        self.__board_size = 6
        self.__batch: np.ndarray = None

    def set_board_size(self, board_size: int):
        self.__board_size = board_size

    def set_batch_size(self, batch_size: int):
        self.__batch = np.empty(shape=(batch_size, self.__board_size, self.__board_size, NN_NUM_CHANNEL))

    def sample_moves(self, positions: list[Position]) -> list[int]:
        batch = self.__batch
        for pos, x in zip(positions, batch):
            position_to_input(pos, x)

        moves = []
        p_logits = self.__net.predict_logit(batch)
        for pos, p_logit in zip(positions, p_logits):
            next_moves = list(pos.get_next_moves())

            if len(next_moves) == 0:
                moves.append(pos.PASS_COORD)
                continue

            policy = tf.nn.softmax(p_logit[next_moves]).numpy().astype(np.float64)
            # policyを64bitにした後に再度, 和が1になるように正規化しないとnp.random.choiceでエラーが出ることがある.
            np.divide(policy, np.sum(policy), out=policy)
            moves.append(np.random.choice(next_moves, p=policy).item())

        return moves

def score_to_reward(score: int) -> float:
    if score == 0:
        return 0.0
    return 1.0 if score > 0 else -1.0

def exec_episodes(config: ReinforceConfig, shared: SharedStorage) -> list[tuple[Position, int, float]]:
    board_size, batch_size = config.board_size, config.num_episode_in_batch // 2
    target_policy = NetworkPolicy(shared.policy_net)
    opponent_policy = shared.policy_pool.sample()

    target_policy.set_board_size(board_size)
    opponent_policy.set_board_size(board_size)

    target_policy.set_batch_size(batch_size)
    opponent_policy.set_batch_size(batch_size)

    episode_id = shared.episode_count + 1
    print(f"episodes: {episode_id} to {episode_id + config.num_episode_in_batch - 1}")

    train_data = []
    for i in range(2):
        if i == 0:
            target_color = DiscColor.BLACK
            black_policy, white_policy = target_policy, opponent_policy
        else:
            target_color = DiscColor.WHITE
            black_policy, white_policy = opponent_policy, target_policy

        positions = [Position(config.board_size, trans_prob=config.trans_prob) for _ in range(batch_size)]
        histories = [[] for _ in range(batch_size)]
        pass_counts = [0] * batch_size
        move_count = 0
        while True:
            actor_policy = black_policy if move_count % 2 == 0 else white_policy
            moves = actor_policy.sample_moves(positions)

            terminal_pos_count = 0
            for i, (pos, move_coord, history) in enumerate(zip(positions, moves, histories)):
                if pass_counts[i] == 2:
                    terminal_pos_count += 1
                    continue

                if move_coord == pos.PASS_COORD:
                    pos.do_pass()
                    pass_counts[i] += 1
                else:
                    pos.do_move(pos.get_move(move_coord))
                    pass_counts[i] = 0

                if actor_policy is target_policy:
                    history.append([pos.copy(), move_coord])

            if terminal_pos_count == len(positions):
                break

            move_count += 1

        for pos, history in zip(positions, histories):
            reward = score_to_reward(pos.get_score_from(target_color))
            for pos, move_coord in history:
                train_data.append((pos, move_coord, reward))

    shared.episode_count += config.num_episode_in_batch

    return train_data


def train(config: ReinforceConfig, shared: SharedStorage, train_data: list[tuple[Position, int, float]]):
    EPSILON = 1.0e-10

    batch_size = len(train_data)
    board_size = config.board_size
    policy_net = shared.policy_net
    old_policy_net = policy_net.copy()

    batch = np.empty(shape=(batch_size, board_size, board_size, NN_NUM_CHANNEL)).astype(np.float32)
    rewards = np.empty(shape=(batch_size, 1))

    move_coords = []
    for i, (pos, move_coord, reward) in enumerate(train_data):
        position_to_input(pos, batch[i])
        rewards[i][0] = reward
        move_coords.append(move_coord)

    masks = tf.one_hot(move_coords, board_size ** 2 + 1)
    with tf.GradientTape() as tape:
        policy = tf.nn.softmax(policy_net.call(batch))
        old_policy = tf.nn.softmax(old_policy_net.call(batch))
        policy_ratio = policy / (old_policy + EPSILON)
        policy_ratio = tf.reduce_sum(-masks * policy_ratio, axis=1)
        clipped_policy_ratio = tf.clip_by_value(policy_ratio, 1.0 - config.ppo_clip_range, 1.0 + config.ppo_clip_range)

        non_clipped_loss = tf.reduce_mean(masks * policy_ratio * rewards)
        clipped_loss = tf.reduce_mean(masks * clipped_policy_ratio, axis=1)
        loss = -tf.minimum(non_clipped_loss, clipped_loss)

        entropy = -tf.reduce_sum(policy * tf.math.log(policy + EPSILON), axis=1)
        loss -= tf.reduce_mean(entropy)
    
    grads = tape.gradient(loss, policy_net.weights)
    policy_net.apply_gradients(grads)

    reward_mean = np.mean(rewards).item()
    shared.reward_histroy.append(reward_mean)

    print(f"loss: {loss.numpy().item()}")
    print(f"reward(win_rate): {(reward_mean + 1.0) * 50.0:.2f}%\n")

    shared.train_count += 1


def main(config: ReinforceConfig, policy_model_path: str = None):
    shared = SharedStorage(config)
    
    if policy_model_path is not None:
        shared.policy_net = PolicyNetwork(model_path=policy_model_path)
        shared.policy_pool.add(NetworkPolicy(shared.policy_net.copy()))

    while shared.train_count < config.train_steps:
        # エピソードの実行
        train_data = exec_episodes(config, shared)

        if shared.train_count != 0 and shared.train_count % config.checkpoint_interval == 0:
            path = config.policy_model_path.format(shared.save_count)
            shared.policy_net.save(path)
            print(f"Info: PolicyNetwork has been saved at \"{path}\"")
            shared.save_count += 1

            with open(config.loss_path, 'w') as file:
                file.write(str(shared.reward_histroy))
            print(f"Info: loss histroy has been saved at \"{config.loss_path}\"")

            shared.policy_pool.add(NetworkPolicy(shared.policy_net.copy()))
        
        train(config, shared, train_data)
        tf.keras.backend.clear_session()

    shared.policy_net.save(config.policy_model_path.format("final"))


if __name__ == "__main__":
    main(ReinforceConfig())
        








