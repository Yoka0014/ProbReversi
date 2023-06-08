import math
import random
import gc

import numpy as np

from prob_reversi import Position, DiscColor, Move


class Node:
    __object_count = 0

    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.move_coords: list[int]     # 着手座標
        self.moves: list[tuple[Move, Move]] = None     # list[(成功着手, 失敗着手)]

        # 子ノード関連の情報.
        self.child_visit_counts: np.ndarray = None
        self.child_value_sums: np.ndarray = None
        self.child_nodes: list[tuple[Node, Node]] = None     # list[[成功した局面のノード, 失敗した局面のノード]]

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

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count

    def init_child_nodes(self):
        """
        子ノードオブジェクトのリストを初期化する.
        """
        [None] * len(self.moves)

    def create_move(self, pos: Position, idx):
        """
        idx番目の成功着手と失敗着手を生成する.
        """
        coord = self.move_coords[idx]
        self.moves[idx] = (pos.get_player_move(coord), pos.get_opponent_move(coord))

    def create_child_node(self, idx):
        """
        idx番目の子ノードを実体化する.
        """
        child = self.child_nodes[idx] = (Node(), Node())
        return child

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


class MoveEval:
    """
    探索の結果得られた着手の価値.
    """

    def __init__(self):
        self.coord: int     # 着手座標
        self.effort: float  # この着手に費やされた探索の割合
        self.playout_count: int     # この着手に費やされたプレイアウト回数
        self.action_value: float    # この着手の行動価値


class SearchResult:
    """
    探索結果
    """

    def __init__(self):
        root_value: MoveEval    # 初期局面の価値
        move_values: list[MoveEval]     # 候補手の価値


class UCTConfig:
    def __init__(self):
        self.expansion_threshold = 40   # ノードの展開閾値
        self.ucb_factor = 1.41421356    # UCB1のバイアス項の強さを決める定数(デフォルト値は理論値であるsqrt(2))
        self.node_num_limit = 10 ** 6   # ノードオブジェクトの上限値


class UCT:
    """
    Upper Confidence Bound applied to trees
    """
    # ルートノード直下のノードのFPU(First Play Urgency)
    # FPUは未訪問ノードの行動価値の初期値. ルートノード直下以外の子ノードは, 親ノードの価値をFPUとして用いる.
    __ROOT_FPU = 1.0

    def __init__(self, config: UCTConfig):
        self.__EXPANSION_THRESHOLD = config.expansion_threshold
        self.__UCB_FACTOR = config.ucb_factor
        self.__NODE_NUM_LIMIT = config.node_num_limit

        self.__playout_count = 0
        self.__search_start_ms = 0
        self.__search_end_ms = 0

        self.__root_pos: Position = None
        self.__root: Node = None

    @property
    def search_ellapsed_ms(self) -> int:
        self.__search_end_ms - self.__search_start_ms

    @property
    def playout_count(self) -> int:
        return self.__playout_count

    @property
    def pps(self) -> float:
        """
        playout per second
        """
        return self.__playout_count / self.search_ellapsed_ms

    def set_root_pos(self, pos: Position):
        self.__root_pos = pos
        gc.collect()
        self.__root = Node()
        self.__init_root_child_nodes()

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
            if root.child_visit_counts[i] == 0:
                root.create_move(pos, i)

            if root.child_nodes[i] is None:
                root.create_child_node(i)

    def __visit_root_node(self, pos: Position):
        node = self.__root
        child_idx = self.__select_root_child_node()
        if node.visit_count <= self.__EXPANSION_THRESHOLD:
            if node.child_visit_counts[child_idx]:
                move_coord = node.move_coords[child_idx]
                moves = node.moves[child_idx] = (pos.get_player_move(move_coord), pos.get_opponent_move(move_coord))

            pos.do_move(moves[0 if random.random() < pos.TRANS_PROB[move_coord] else 1])
            self.__update_stats(node, child_idx, -self.__playout(pos))
        else:
            move_idx = 0 if random.random() < pos.TRANS_PROB[move_coord] else 1
            pos.do_move(node.moves[move_idx])
            self.__update_stats(node, child_idx, -self.__visit_node(pos, node.child_nodes[child_idx][move_idx]))

    def __visit_node(self, pos: Position, node: Node, after_pass=False):
        if not node.is_expanded:
            node.expand(pos)

        value = 0.0
        if node.move_coords[0] == pos.PASS_COORD:
            if after_pass:  # パスが2連続 -> 終局
                score = pos.get_score()
                if score == 0:
                    value = 0.5
                else:
                    value = 1.0 if score > 0 else 0.0
            else:   # パスノードはさらに1手先を読んで評価
                if node.child_nodes is None:
                    node.init_child_nodes()
                    node.create_child_node(0)
                
                pos.do_pass()
                value = self.__visit_node(pos, node.child_nodes[0][0], after_pass=True)

            self.__update_stats(node, 0, value)
            return 1.0 - value
        
        child_idx = self.__select_child_node(node)
        if node.visit_count <= self.__EXPANSION_THRESHOLD:
            if node.child_visit_counts[child_idx] == 0:
                move_coord = node.move_coords[child_idx]
                moves = node.moves[child_idx] = (pos.get_player_move(move_coord), pos.get_opponent_move(move_coord))

            pos.do_move(moves[0 if random.random() < pos.TRANS_PROB[move_coord] else 1])
            value = 1.0 - self.__playout(pos)
            self.__update_stats(node, child_idx, value)
            return 1.0 - value
        
        if node.child_nodes is None:
            node.init_child_nodes()

        if node.child_nodes[child_idx] is None:
            node.create_child_node(child_idx)

        move_idx = 0 if random.random() < pos.TRANS_PROB[move_coord] else 1
        pos.do_move(node.moves[move_idx])
        self.__update_stats(node, child_idx, self.__visit_node(pos, node.child_nodes[child_idx][move_idx]))
        return 1.0 - value


    def __select_root_child_node(self) -> np.intp:
        """
        ルートノード直下の子ノードを選択する.
        """
        parent = self.__root
        if parent.visit_count == 0:
            default_u = 0.0
        else:
            log_sum = math.log(parent.visit_count)
            default_u = math.sqrt(log_sum)

        # 行動価値の計算(未訪問ノードはself.__ROOT_FPUで初期化)
        q = np.divide(parent.child_value_sums, parent.child_visit_counts,
                      out=np.full(parent.num_child, self.__ROOT_FPU, np.float32), where=parent.child_visit_counts != 0)

        # バイアス項の計算
        u = np.divide(log_sum / parent.child_visit_counts,
                      out=np.full(parent.num_child, default_u, np.float32), where=parent.child_visit_counts != 0)
        np.sqrt(u, out=u)

        return np.argmax(q + self.__UCB_FACTOR * u)

    def __select_child_node(self, parent: Node) -> np.intp:
        """
        子ノードを選択する.
        """
        if parent.visit_count == 0:
            default_u = 0.0
        else:
            log_sum = math.log(parent.visit_count)
            default_u = math.sqrt(log_sum)

        # 未訪問ノードの価値は親ノードの価値で初期化
        fpu = parent.value

        # 行動価値の計算
        q = np.divide(parent.child_value_sums, parent.child_visit_counts,
                      out=np.full(parent.num_child, fpu, np.float32), where=parent.child_visit_counts != 0)

        # バイアス項の計算
        u = np.divide(log_sum / parent.child_visit_counts,
                      out=np.full(parent.num_child, default_u, np.float32), where=parent.child_visit_counts != 0)
        np.sqrt(u, out=u)

        return np.argmax(q + self.__UCB_FACTOR * u)

    def __playout(self, pos: Position):
        player = pos.side_to_move
        pass_count = 0
        while pass_count != 2:
            coord = pos.sample_next_move()
            if coord == pos.PASS_COORD:
                pos.do_pass()
                pass_count += 1
                continue

            pass_count = 0
            move = pos.get_move(coord)
            pos.do_move(move)

        score = pos.get_score_from(player)
        if score == 0:
            return 0.5
        return 1.0 if score > 0 else 0.0

    def __update_stats(self, parent: Node, child_idx: int, value: float):
        parent.visit_count += 1
        parent.child_visit_counts[child_idx] += 1
        parent.value_sum += value
        parent.child_value_sums[child_idx] += value
