"""
確率リバーシの盤面関連.
一部, C++で実装(cppディレクトリ内のself.__helper.hと__self.__helper.cppを参照).
"""

from typing import List, Generator, Tuple
from enum import IntEnum
import random
import copy

import prob_reversi_helper  # C++で書いた着手可能位置計算と裏返る石を求める計算を行うヘルパー関数を使用するためのモジュール.

class DiscColor(IntEnum):
    BLACK = 0
    WHITE = 1
    NULL = 2


class Player(IntEnum):
    CURRENT = 0 
    OPPONENT = 1
    NULL = 2


def to_opponent_color(color: DiscColor) -> DiscColor:
    return DiscColor(color ^ DiscColor.WHITE)   # Disc.NULL を引数で与えた際の動作は未定義.


class Move:
    """
    着手を表現するクラス.

    Attributes
    ----------
    player: Player
        どちらのプレイヤーの石が着手されるか.
    coord: int
        着手位置の座標.
    flip: int
        裏返る石の配置.
    """
    def __init__(self, player=Player.NULL, coord=0, flip=0):
        self.player = player
        self.coord = coord   
        self.flip = flip


class Position:
    """
    リバーシの局面を表現するクラス.

    Attributes
    ----------
    SIZE: int
        盤面のサイズ(4x4ならSIZE == 4).
    SQUARE_NUM: int
        マス目の数.
    PASS_COORD: int
        パスを表す座標.
    TRANS_PROB: List[float]
        各マス目の着手確率. リストの各要素がそのマスに自分の石を置ける確率に相当する.

    Note
    ----
    リバーシの盤面はビットボードというデータ構造で表現している.
    ビットボードでは, SQUARE_NUM bitのビット列を2つ用いて, 黒石と白石の配置を管理する.
    リストなどで実装するよりもデータ量が削減でき, ビット演算を用いれば複数のマス目を同時に処理することもできる.
    """
    def __init__(self, size, trans_prob = None):
        self.__helper = prob_reversi_helper.Helper(size)

        self.SIZE = size
        self.SQUARE_NUM = size * size
        self.PASS_COORD = self.SQUARE_NUM
        self.__VALID_BITS_MASK = (1 << self.SQUARE_NUM) - 1
        self.__side_to_move = DiscColor.BLACK
        self.__opponent_color = DiscColor.WHITE
        self.__player = 0   # 現在の手番の石の配置.
        self.__opponent = 0 # 相手の石の配置.
        self.__rand = random.Random()
        self.clear()

        if trans_prob is None:
            self.TRANS_PROB = [1.0 for _ in range(self.SQUARE_NUM)]
        else:
            if len(trans_prob) != self.SQUARE_NUM:
                raise ValueError("The length of trans_prob must be same as SQUARE_NUM.")
            
            self.TRANS_PROB = []
            for prob in trans_prob:
                if prob < 0.0 or prob > 1.0:
                    raise ValueError("Transition probability must be in [0.0, 1.0].")
                self.TRANS_PROB.append(prob)

    def clear(self):
        """
        盤面をクリアして, 初期配置にする.
        """
        # 中央に石をクロス配置する.
        size = self.SIZE
        x = size // 2 - 1
        self.__player = 0   
        self.__opponent = 0 
        self.put_player_disc_at((x + 1) + x * size)
        self.put_player_disc_at(x + (x + 1) * size)
        self.put_opponent_disc_at(x + x * size)
        self.put_opponent_disc_at((x + 1) + (x + 1) * size)
        self.__side_to_move = DiscColor.BLACK
        self.__opponent_color = DiscColor.WHITE

    @property
    def bitboard(self) -> Tuple[int, int]:
        """
        現在のビットボードを取得する.
        """
        return self.__player, self.__opponent

    @property
    def side_to_move(self) -> DiscColor:
        """
        現在の手番の石の色を返す.
        """
        return self.__side_to_move
    
    @property
    def opponent_color(self) -> DiscColor:
        """
        現在の手番ではないプレイヤーの石の色を返す.
        """
        return self.__opponent_color
    
    @property
    def empty_square_count(self) -> int:
        return (~(self.__player | self.__opponent) & self.__VALID_BITS_MASK).bit_count()
    
    @property
    def player_disc_count(self) -> int:
        """
        現在の手番の石の数を返す.
        """
        return self.__player.bit_count()

    @property
    def opponent_disc_count(self) -> int:
        """
        相手の石の数を返す.
        """
        return self.__opponent.bit_count()

    @property
    def disc_count(self) -> int:
        """
        全ての石の数を返す.
        """
        return (self.__player | self.__opponent).bit_count()

    def set_state(self, player: int, opponent: int, side_to_move: DiscColor):
        """
        局面の状態を設定する.
        """
        self.__player, self.__opponent = player, opponent
        self.__side_to_move = side_to_move

    def copy(self):
        pos = Position(self.SIZE)
        self.copy_to(pos)
        return pos
    
    def copy_to(self, dest, copy_trans_prob=True):
        dest.__player, dest.__opponent = self.__player, self.__opponent
        dest.__side_to_move, dest.__opponent_color = self.__side_to_move, self.__opponent_color
        if copy_trans_prob:
            dest.TRANS_PROB = copy.copy(self.TRANS_PROB)

    def get_disc_count_of(self, color: DiscColor) -> int:
        """
        指定された色の石の数を返す.
        """
        return self.__player.bit_count() if self.__side_to_move == color else self.__opponent.bit_count()
    
    def get_square_color_at(self, coord: int):
        """
        指定された座標のマス目に何色の石が配置されているか取得する.
        """
        owner = self.get_square_owner_at(self, coord)
        if owner == Player.NULL:
            return DiscColor.NULL
        return self.__side_to_move if owner == Player.CURRENT else self.__opponent_color

    def get_square_owner_at(self, coord: int) -> Player:
        """
        指定された座標のマス目に現在の手番と相手のどちらのプレイヤーの石が配置されているか取得する.

        Parameters
        ----------
        coord: int
            マス目の座標.

        Returns
        -------
        owner: Owner
            マス目に配置されている石の所有者. 現在の手番の石が配置されている場合は Owner.CURRENT, 相手の石の場合は Owner.OPPONENT, 石が配置されていない場合は Owner.NULL.
        """
        ret = 2 - 2 * ((self.__player // (1 << coord)) & 1) - ((self.__opponent // (1 << coord)) & 1)
        return Player(ret)
    
    def parse_coord(self, coord_str: str) -> int:
        """
        文字列で表現された盤面の座標を整数値に変換する.
        """
        coord_str = coord_str.strip().lower()

        if coord_str == "pass" or coord_str == "pa":
            return self.PASS_COORD

        if coord_str[0] < 'a' or coord_str[0] > chr(ord('a') + self.SIZE - 1):
            raise ValueError(f"Coordinate {coord_str} is invalid.")
        
        x = ord(coord_str[0]) - ord('a')
        y = int(coord_str[1:]) - 1

        if y < 0 or y >= self.SIZE:
            raise ValueError(f"Coordinate {coord_str} is invalid.")

        return x + y * self.SIZE
    
    def convert_coord_to_str(self, coord: int) -> str:
        if coord == self.PASS_COORD:
            return "Pass"
        
        x, y = coord % self.SIZE, coord // self.SIZE
        return f"{chr(ord('A') + x)}{y + 1}"
    
    def __eq__(self, right: object) -> bool:
        if type(right) is not type(self):
            return False
        return self.__side_to_move == right.__side_to_move and self.__player == right.__player and self.__opponent == right.__opponent and self.TRANS_PROB == right.TRANS_PROB
    
    def __str__(self) -> str:
        s = "  "
        for i in range(self.SIZE):
            s += f"{chr(ord('A') + i)} "

        p, o = self.__player, self.__opponent
        side_to_move = self.__side_to_move
        mask = 1
        for y in range(self.SIZE):
            s += f"\n{y + 1} "
            for x in range(self.SIZE):
                if p & mask:
                    if side_to_move == DiscColor.BLACK:
                        s += "* "
                    else:
                        s += "O "
                elif o & mask:
                    if side_to_move == DiscColor.BLACK:
                        s += "O "
                    else:
                        s += "* "
                else:
                    s += "- "
                
                mask <<= 1
        
        return s

    def put_player_disc_at(self, coord: int):
        """
        指定された座標のマス目に現在の手番の石を配置する. ただし, 石を配置するだけで裏返さない.
        """
        bit = 1 << coord
        self.__player |= bit

        if self.__opponent & bit:   # 2つの石が1つのマス目に同時に存在している場合は, 他方を排除する.
            self.__opponent ^= bit

    def put_opponent_disc_at(self, coord: int):
        """
        指定された座標のマス目に相手の石を配置する. ただし, 石を配置するだけで裏返さない.
        """
        bit = 1 <<  coord
        self.__opponent |= bit

        if self.__player & bit:   # 2つの石が1つのマス目に同時に存在している場合は, 他方を排除する.
            self.__player ^= bit

    def remove_disc_at(self, coord: int):
        """
        指定された座標のマス目に配置されている石を取り除く. 
        """
        bit = 1 << coord
        if self.__player & bit:
            self.__player ^= bit

        if self.__opponent & bit:
            self.__opponent ^= bit

    def is_gameover(self) -> bool:
        """
        終局しているかどうかを返す.
        """
        p, o = self.__player, self.__opponent
        return self.__helper.calc_mobility(p, o).bit_count() == 0 and self.__helper.calc_mobility(o, p).bit_count() == 0
    
    def get_score(self) -> int:
        """
        現在の手番からみた石差を返す.
        """
        return self.__player.bit_count() - self.__opponent.bit_count()

    def can_pass(self) -> bool:
        """
        パスが可能な局面かどうかを返す.
        """
        return self.__helper.calc_mobility(self.__player, self.__opponent).bit_count() == 0

    def do_pass(self):
        """
        着手を行わずに手番を交代する.
        """
        self.__side_to_move, self.__opponent_color = self.__opponent_color, self.__side_to_move
        self.__player, self.__opponent = self.__opponent, self.__player

    def get_next_moves(self) -> Generator[int, None, None]:
        """
        着手可能な位置を取得する.
        """
        mobility = self.__helper.calc_mobility(self.__player, self.__opponent)
        while mobility:
            coord = (mobility & -mobility).bit_length() - 1  
            yield coord
            mobility &= (mobility - 1)

    def sample_next_move(self) -> int:
        """
        次の着手位置をランダムにサンプリングする.
        """
        mobility = self.__helper.calc_mobility(self.__player, self.__opponent)

        move_num = mobility.bit_count()
        if move_num == 0:
            return self.PASS_COORD
        
        idx = random.randint(0, move_num - 1)
        i = 0
        while mobility:
            coord = (mobility & -mobility).bit_length() - 1  
            if i == idx:
                return coord
            mobility &= (mobility - 1)
            i += 1
            
    def get_move(self, coord: int) -> Move:
        """
        与えられた位置に着手する場合のMoveオブジェクトを取得する.
        Moveオブジェクトの内容は着手確率(TRANS_PROB)よって変わる.

        Note
        ----
        coord == self.PASS_COORDの場合は未定義. do_pass関数を用いること. 
        """
        # 着手確率に従って, 手番の石が配置されるか, 相手の石が配置されるか, 石が配置されないかを決める.
        prob = self.TRANS_PROB[coord]
        rand = self.__rand.random()

        if rand < prob:  # 手番の石を置ける.
            return Move(Player.CURRENT, coord, self.__helper.calc_flip_discs(self.__player, self.__opponent, coord))
        
        return Move(Player.OPPONENT, coord, self.__helper.calc_flip_discs(self.__opponent, self.__player, coord))  # 相手に石を置かれる.
    
    def get_player_move(self, coord: int) -> Move:
        """
        与えられた位置に着手する場合の手番側のMoveオブジェクトを取得する.

        Note
        ----
        coord == self.PASS_COORDの場合は未定義. do_pass関数を用いること. 
        """
        return Move(Player.CURRENT, coord, self.__helper.calc_flip_discs(self.__player, self.__opponent, coord))
    
    def get_opponent_move(self, coord: int) -> Move:
        """
        与えられた位置に着手する場合の相手側のMoveオブジェクトを取得する.

        Note
        ----
        coord == self.PASS_COORDの場合は未定義. do_pass関数を用いること. 
        """
        return Move(Player.OPPONENT, coord, self.__helper.calc_flip_discs(self.__opponent, self.__player, coord))
    
    def do_move(self, move: Move):
        """
        与えられた着手に基づき, 現在の手番の石で盤面を更新する. 

        Note
        ----
        高速化のため, 合法手チェックはしない.
        """
        if move.player == Player.CURRENT:
            player = self.__player
            self.__player = self.__opponent ^ move.flip
            self.__opponent = player | (1 << move.coord) | move.flip
            self.__side_to_move, self.__opponent_color = self.__opponent_color, self.__side_to_move
        else:
            opponent = self.__opponent
            self.__opponent = self.__player ^ move.flip
            self.__player = opponent | (1 << move.coord) | move.flip
            self.__side_to_move, self.__opponent_color = self.__opponent_color, self.__side_to_move

    def do_move_at(self, coord: int) -> bool:
        """
        与えられた着手位置に基づき, 盤面を更新する.

        Returns
        -------
        legal: bool
            与えられた着手位置が有効であればTrue, そうでなければFalse.
        """
        coord_bit = 1 << coord

        if coord == self.PASS_COORD:    
            if self.can_pass() != 0:    # パスできないのにパスをしようとした.
                return False
            
        if not (self.__helper.calc_mobility(self.__player, self.__opponent) & coord_bit):    # 着手できない場所に着手しようとした.
            return False
        
        self.do_move(self.get_move(coord))
        return True