import math
import random
import gc
import time
from enum import Enum

import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from prob_reversi import Position, DiscColor, Move
from dualnet import DN_NUM_CHANNEL, position_to_input


class Node:
    __object_count = 0

    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.policy: np.ndarray = None
        self.value: float = None
        self.move_coords: list[int] = None    # 着手座標
        self.moves: list[tuple[Move, Move]] = None     # list[(成功着手, 失敗着手)]

        # 子ノード関連の情報.
        self.child_visit_counts: np.ndarray = None
        self.child_value_sums: np.ndarray = None
        self.child_nodes: list[list[Node, Node]] = None     # list[[成功した局面のノード, 失敗した局面のノード]]

        Node.__object_count += 1

    def __del__(self):
        Node.__object_count -= 1

    @staticmethod
    def get_object_count() -> int:
        return Node.__object_count

    @property
    def is_expanded(self) -> bool:
        return self.child_visit_counts is not None

    @property
    def num_child(self) -> int:
        return len(self.move_coords)

    def init_child_nodes(self):
        """
        子ノードオブジェクトのリストを初期化する.
        """
        self.child_nodes = [None] * len(self.moves)

    def expand(self, pos: Position):
        """
        合法手の数だけ子ノードを展開(厳密には子ノードに至る辺を展開).
        """
        self.move_coords = list(pos.get_next_moves())
        num_child = len(self.move_coords)
        self.moves = [None] * num_child

        if num_child != 0:
            self.child_visit_counts = np.zeros(num_child, dtype=np.uint32)
            self.child_value_sums = np.zeros(num_child, dtype=np.float32)
        else:
            self.child_visit_counts = np.zeros(1, dtype=np.uint32)
            self.child_value_sums = np.zeros(1, dtype=np.float32)
            self.move_coords = [pos.PASS_COORD]
            self.moves = [None]


class MoveEval:
    """
    探索の結果得られた着手の価値.
    """

    def __init__(self):
        self.coord = 0             # 着手座標
        self.policy_prob = 0.0     # NNが出力した方策の事前確率
        self.effort = 0.0          # この着手に費やされた探索の割合
        self.playout_count = 0     # この着手に費やされたプレイアウト回数
        self.action_value = 0.0    # この着手の行動価値

    def copy(self):
        copied = MoveEval()
        copied.coord = self.coord
        copied.policy_prob = self.policy_prob
        copied.effort = self.effort
        copied.playout_count = self.playout_count
        copied.action_value = self.action_value
        return copied


class SearchResult:
    """
    探索結果
    """

    def __init__(self):
        self.root_value: MoveEval = MoveEval()   # 初期局面の価値
        self.move_evals: list[MoveEval] = []     # 候補手の価値
        self.ellapsed_ms = 0                     # 探索に要した時間[ms]

    def copy(self):
        copied = SearchResult()
        copied.root_value = self.root_value
        copied.move_evals = [e.copy() for e in self.move_evals]
        copied.ellapsed_ms = self.ellapsed_ms
        return copied


class UCTConfig:
    def __init__(self):
        self.model_path: str = None
        self.c_init = 1.25
        self.c_base = 19652
        self.reuse_subtree = True       # 可能なら前回の探索結果を再利用する
        self.batch_size = 32    # まとめて評価する局面数
        self.softmax_temperature = 1.0


class TrajectoryItem:
    def __init__(self, node: Node, child_idx: int, move_idx: int):
        self.node = node
        self.child_idx = child_idx
        self.move_idx = move_idx


class VisitResult(Enum):
    QUEUING = 0
    DISCARDED = 1
    TERMINAL = 2


class UCT:
    """
    Upper Confidence Bound applied to trees
    """

    """
    ルートノード直下のノードのFPU(First Play Urgency)
    FPUは未訪問ノードの行動価値. ルートノード直下以外の子ノードは, 親ノードの価値をFPUとして用いる.

    Note:
        ルートノード直下の未訪問ノードは全て勝ちと見做す. そうすれば, 1手先の全ての子ノードは初期に少なくとも1回はプレイアウトされる.

        ルートノード直下以外の未訪問ノードは, 親ノードの価値で初期化する. 
        そうすれば, 親ノードよりも価値の高い子ノードが見つかれば, しばらくそのノードが選ばれ続ける.
    """
    __ROOT_FPU = 1.0
    __VIRTUAL_LOSS = 1

    def __init__(self, config: UCTConfig):
        self.__C_BASE = config.c_base
        self.__C_INIT = config.c_init
        self.__MODEL: Model = load_model(config.model_path, custom_objects={"softmax_cross_entropy_with_logits_v2": tf.nn.softmax_cross_entropy_with_logits})
        self.__BATCH_SIZE = config.batch_size
        self.__SOFTMAX_TEMPERATURE = config.softmax_temperature

        self.__batch: np.ndarray = None
        self.__predict_queue: list[Node] = []

        self.__playout_count = 0
        self.__search_start_ms = 0
        self.__search_end_ms = 0

        self.__root_pos: Position = None
        self.__root: Node = None

    @property
    def search_ellapsed_ms(self) -> int:
        return self.__search_end_ms - self.__search_start_ms

    @property
    def playout_count(self) -> int:
        return self.__playout_count

    @property
    def pps(self) -> float:
        """
        playout per second
        """
        return self.__playout_count / (self.search_ellapsed_ms * 1.0e-3)

    def set_root_pos(self, pos: Position):
        prev_root_pos = self.__root_pos
        self.__root_pos = pos.copy()

        # 前回の探索結果を再利用できるか確認.
        if self.__root is not None:
            for i in range(self.__root.num_child):
                move = self.__root.moves[i]

                for move_idx in range(2):
                    m = move[move_idx]
                    if m is None:
                        continue

                    next_pos = prev_root_pos.copy()
                    if m.coord != next_pos.PASS_COORD:
                        next_pos.do_move(move[move_idx])
                    else:
                        next_pos.do_pass()

                    if next_pos == pos and self.__root.child_nodes[i][move_idx] is not None:  # 再利用可能
                        self.__root = self.__root.child_nodes[i][move_idx]
                        self.__init_root_child_nodes()
                        gc.collect()
                        return

        self.__root = Node()
        self.__batch = np.empty(shape=(self.__BATCH_SIZE, pos.SIZE, pos.SIZE, DN_NUM_CHANNEL))
        self.__init_root_child_nodes()
        gc.collect()

    def search(self, num_playouts: int) -> SearchResult:
        root_pos = self.__root_pos
        pos = Position(root_pos.SIZE, root_pos.TRANS_PROB)
        trajectories: list[list[TrajectoryItem]] = []
        trajectories_discarded: list[list[TrajectoryItem]] = []

        self.__playout_count = 0
        self.__search_start_ms = int(time.perf_counter() * 1000.0)

        while self.__playout_count < num_playouts:
            trajectories.clear()
            trajectories_discarded.clear()
            self.__predict_queue.clear()
            self.__current_batch_idx = 0

            for i in range(self.__BATCH_SIZE):
                root_pos.copy_to(pos, copy_trans_prob=False)

                trajectories.append([])
                result = self.__visit_root_node(pos, trajectories[-1])

                if result != VisitResult.DISCARDED:
                    self.__playout_count += 1   # 評価待ちノードもプレイアウト数に加算されてしまうが許容する
                else:
                    trajectories_discarded.append(trajectories[-1])

                    # 頻繁に評価待ちノードに訪問する場合はキューが満杯になる前に推論する
                    if len(trajectories_discarded) > self.__BATCH_SIZE // 2:
                        trajectories.pop()
                        break

                if result != VisitResult.QUEUING:
                    trajectories.pop()

            if len(trajectories) > 0:
                self.__predict()

            for trajectory in trajectories_discarded:
                self.__remove_virtual_loss(trajectory)

            for trajectory in trajectories:
                self.__backup(trajectory)

        self.__search_end_ms = int(time.perf_counter() * 1000.0)
        tf.keras.backend.clear_session()
        return self.collect_search_result()

    def collect_search_result(self) -> SearchResult:
        root = self.__root
        result = SearchResult()
        result.root_value.playout_count = root.visit_count
        result.root_value.effort = 1.0
        result.root_value.action_value = root.value_sum / root.visit_count
        for i in range(root.num_child):
            eval = MoveEval()
            eval.coord = root.move_coords[i]
            eval.policy_prob = root.policy[i].item()
            eval.playout_count = root.child_visit_counts[i]
            eval.effort = eval.playout_count / root.visit_count
            eval.action_value = root.child_value_sums[i] / root.child_visit_counts[i]
            result.move_evals.append(eval)
        return result

    def get_search_result_str(self) -> str:
        res = self.collect_search_result()
        s = []
        s.append(f"ellpased={self.search_ellapsed_ms}[ms]\t{self.playout_count}[playouts]\t{self.pps:.2f}[pps]\n")
        s.append(f"win_rate={res.root_value.action_value * 100.0:.2f}%\n")
        s.append("|move|policy|effort|playouts|win_rate|\n")

        for eval in sorted(res.move_evals, key=lambda e: 1.0 - e.effort):
            s.append(f"| {self.__root_pos.convert_coord_to_str(eval.coord)} ")

            s.append("|")
            s.append(f"{eval.policy_prob * 100:.2f}%".rjust(6))

            s.append("|")
            s.append(f"{eval.effort * 100:.2f}%".rjust(6))

            s.append("|")
            s.append(str(eval.playout_count).rjust(8))

            s.append("|")
            s.append(f"{eval.action_value * 100:.2f}%".rjust(8))

            s.append("|\n")

        return "".join(s)

    def __init_root_child_nodes(self):
        """
        ルートノード直下の子ノードを初期化する.
        """
        pos = self.__root_pos
        root = self.__root
        if not root.is_expanded:
            self.__root.expand(pos)

        if root.child_nodes is None:
            root.init_child_nodes()

        for i in range(root.num_child):
            coord = root.move_coords[i]
            root.moves[i] = (pos.get_player_move(coord), pos.get_opponent_move(coord))

            if root.child_nodes[i] is None:
                root.child_nodes[i] = [None, None]

        if root.policy is None or root.value is None:
            x = position_to_input(pos)
            x = x[np.newaxis, :, :, :]
            p_logits, v = self.__MODEL.predict(x, verbose=0)
            root.policy = tf.nn.softmax(p_logits[0][root.move_coords] / self.__SOFTMAX_TEMPERATURE, axis=0).numpy()
            root.value = v[0].item()

    def __visit_root_node(self, pos: Position, trajectory: list[TrajectoryItem]) -> VisitResult | float:
        node = self.__root
        child_idx = self.__select_root_child_node()
        move_coord = node.move_coords[child_idx]
        node.visit_count += UCT.__VIRTUAL_LOSS
        node.child_visit_counts[child_idx] += UCT.__VIRTUAL_LOSS

        move_idx = 0 if random.random() < pos.TRANS_PROB[move_coord] else 1
        pos.do_move(node.moves[child_idx][move_idx])
        trajectory.append(TrajectoryItem(node, child_idx, move_idx))

        child_node = node.child_nodes[child_idx]
        if child_node[move_idx] is None:  # 初訪問
            child_node[move_idx] = Node()
            child_node[move_idx].expand(pos)
            self.__enqueue_node(pos, child_node[move_idx])
            return VisitResult.QUEUING
        elif child_node[move_idx].value is None:    # 2回目の訪問だが評価待ち
            return VisitResult.DISCARDED

        result = self.__visit_node(pos, node.child_nodes[child_idx][move_idx], trajectory)

        if result == VisitResult.QUEUING or result == VisitResult.DISCARDED:
            return result

        self.__update_stats(node, child_idx, result)
        return 1.0 - result

    def __visit_node(self, pos: Position, node: Node, trajectory: list[TrajectoryItem], after_pass=False) -> VisitResult | float:
        if node.move_coords[0] == pos.PASS_COORD:
            # パスノードはさらに1手先を読んで評価
            pos.do_pass()
            trajectory.append(TrajectoryItem(node, 0, 0))

            first_visit = False
            if node.child_nodes is None:
                node.init_child_nodes()
                node.child_nodes[0] = [Node(), None]
                node.child_nodes[0][0].expand(pos)
                first_visit = True

            node.visit_count += UCT.__VIRTUAL_LOSS
            node.child_visit_counts[0] += UCT.__VIRTUAL_LOSS

            child_node = node.child_nodes[0]
            if child_node[0].move_coords[0] == pos.PASS_COORD:  # パスが2連続 -> 終局
                score = pos.get_score()
                if score == 0:
                    reward = 0.5
                else:
                    reward = 0.0 if score > 0 else 1.0
                self.__update_stats(node, 0, reward)
                return 1.0 - reward

            if first_visit:
                self.__enqueue_node(pos, child_node[0])
                return VisitResult.QUEUING
            elif child_node[0].value is None:
                return VisitResult.DISCARDED
            result = self.__visit_node(pos, node.child_nodes[0][0], trajectory, after_pass=True)

            if result == VisitResult.QUEUING or result == VisitResult.DISCARDED:
                return result

            self.__update_stats(node, 0, result)
            return 1.0 - result

        if node.child_nodes is None:
            node.init_child_nodes()

        child_idx = self.__select_child_node(node)
        move_coord = node.move_coords[child_idx]
        if node.child_visit_counts[child_idx] == 0:
            node.moves[child_idx] = [None, None]
            node.child_nodes[child_idx] = [None, None]

        node.visit_count += UCT.__VIRTUAL_LOSS
        node.child_visit_counts[child_idx] += UCT.__VIRTUAL_LOSS

        child_node = node.child_nodes[child_idx]
        move = node.moves[child_idx]
        if random.random() < pos.TRANS_PROB[move_coord]:
            move_idx = 0
            if move[0] is None:
                move[0] = pos.get_player_move(move_coord)
        else:
            move_idx = 1
            if move[1] is None:
                move[1] = pos.get_opponent_move(move_coord)

        pos.do_move(move[move_idx])
        trajectory.append(TrajectoryItem(node, child_idx, move_idx))

        if child_node[move_idx] is None:    # 初訪問
            child_node[move_idx] = Node()
            child_node[move_idx].expand(pos)
            self.__enqueue_node(pos, child_node[move_idx])
            return VisitResult.QUEUING
        elif child_node[move_idx].value is None:    # 2回目の訪問だが評価待ち
            return VisitResult.DISCARDED

        result = self.__visit_node(pos, child_node[move_idx], trajectory)

        if result == VisitResult.QUEUING or result == VisitResult.DISCARDED:
            return result

        self.__update_stats(node, child_idx, result)
        return 1.0 - result

    def __select_root_child_node(self) -> np.intp:
        """
        ルートノード直下の子ノードを選択する.
        """
        parent = self.__root

        # 行動価値の計算(未訪問ノードはself.__ROOT_FPUで初期化)
        q = np.divide(parent.child_value_sums, parent.child_visit_counts,
                      out=np.full(parent.num_child, self.__ROOT_FPU, np.float32), where=parent.child_visit_counts != 0)

        # バイアス項の計算
        if parent.visit_count == 0:
            u = 1.0
        else:
            sqrt_sum = math.sqrt(parent.visit_count)
            u = sqrt_sum / (1.0 + parent.child_visit_counts)

        c_base = self.__C_BASE
        c = math.log((1.0 + parent.visit_count + c_base) / c_base) + self.__C_INIT
        return np.argmax(q + c * parent.policy * u)

    def __select_child_node(self, parent: Node) -> np.intp:
        """
        子ノードを選択する.
        """

        # 未訪問ノードの価値は親ノードの価値で初期化
        fpu = 0.0

        # 行動価値の計算
        q = np.divide(parent.child_value_sums, parent.child_visit_counts,
                      out=np.full(parent.num_child, fpu, np.float32), where=parent.child_visit_counts != 0)

        # バイアス項の計算
        if parent.visit_count == 0:
            u = 1.0
        else:
            sqrt_sum = math.sqrt(parent.visit_count)
            u = sqrt_sum / (1.0 + parent.child_visit_counts)

        c_base = self.__C_BASE
        c = math.log((1.0 + parent.visit_count + c_base) / c_base) + self.__C_INIT
        return np.argmax(q + c * parent.policy * u)

    def __enqueue_node(self, pos: Position, node: Node):
        position_to_input(pos, self.__batch[self.__current_batch_idx])
        self.__predict_queue.append(node)
        self.__current_batch_idx += 1

    def __predict(self):
        p_logits, v = self.__MODEL.predict(self.__batch, batch_size=self.__current_batch_idx, verbose=0)
        for i in range(self.__current_batch_idx):
            node = self.__predict_queue[i]
            node.policy = tf.nn.softmax(p_logits[i][node.move_coords] / self.__SOFTMAX_TEMPERATURE, axis=0).numpy()
            node.value = (v[i].item() + 1.0) * 0.5     # [-1.0, 1.0] -> [0.0, 1.0] に変換

    def __update_stats(self, parent: Node, child_idx: int, value: float):
        parent.visit_count += (1 - UCT.__VIRTUAL_LOSS)
        parent.child_visit_counts[child_idx] += (1 - UCT.__VIRTUAL_LOSS)
        parent.value_sum += value
        parent.child_value_sums[child_idx] += value

    def __remove_virtual_loss(self, trajectory: list[TrajectoryItem]):
        for item in trajectory:
            item.node.visit_count -= UCT.__VIRTUAL_LOSS
            item.node.child_visit_counts[item.child_idx] -= UCT.__VIRTUAL_LOSS

    def __backup(self, trajectory: list[TrajectoryItem]):
        result = None
        for item in reversed(trajectory):
            node = item.node
            child_idx = item.child_idx
            move_idx = item.move_idx
            if result is None:
                result = 1.0 - node.child_nodes[child_idx][move_idx].value
            self.__update_stats(node, child_idx, result)
            result = 1.0 - result
