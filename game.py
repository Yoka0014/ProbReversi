"""
確率リバーシの対局を行うためのコード.
"""
from abc import ABCMeta, abstractproperty, abstractmethod  # 抽象クラスを定義するために必要.
from dataclasses import dataclass
import random

from prob_reversi import DiscColor, Player, Move, Position


class IPlayer(metaclass=ABCMeta):
    """
    プレイヤーが実装するインターフェース.
    """
    @abstractproperty
    def name(self) -> str:
        """
        プレイヤーの名前.
        """
        pass

    @abstractmethod
    def set_position(self, pos: Position):
        """
        局面を設定する.
        """
        pass

    @abstractmethod
    def gen_move(self) -> int:
        """
        着手を決定して返す.
        """
        pass

    @abstractmethod
    def do_move(self, move: Move):
        """
        与えられた着手で局面を更新する.
        """
        pass

    @abstractmethod
    def do_pass(self):
        """
        パスをする.
        """
        pass


class PlayerStats:
    def __init__(self):
        self.win_count = [0, 0]  # (黒番の時の勝利回数, 白番の時の勝利回数)
        self.loss_count = [0, 0]  # (黒番の時の敗北回数, 白番の時の敗北回数)
        self.draw_count = [0, 0]  # (黒番の時の引き分け回数, 白番の時の引き分け回数)

    @property
    def game_count(self) -> list[int]:
        """
        (黒番の時の対局回数, 白番の時の対局回数)
        """
        return [w + l + d for w, l, d in zip(self.win_count, self.loss_count, self.draw_count)]

    @property
    def total_win_count(self) -> int:
        """
        勝利回数.
        """
        return sum(self.win_count)

    @property
    def total_loss_count(self) -> int:
        """
        敗北回数
        """
        return sum(self.loss_count)

    @property
    def total_draw_count(self) -> int:
        """
        引き分け回数
        """
        return sum(self.draw_count)

    @property
    def total_game_count(self) -> int:
        """
        対局回数
        """
        return sum(self.game_count)

    @property
    def win_rate(self) -> list[float]:
        """
        (黒番の時の勝率, 白番の時の勝率)    
        """
        return [(w + d * 0.5) / n for w, d, n in zip(self.win_count, self.draw_count, self.game_count)]   # 引き分けは0.5勝扱い.

    @property
    def total_win_rate(self) -> float:
        """
        勝率
        """
        return (self.total_win_count + self.total_draw_count * 0.5) / self.total_game_count    # 引き分けは0.5勝扱い.


@dataclass
class PlayerData:
    stats: PlayerStats
    player: IPlayer


def start(player_0: IPlayer, player_1: IPlayer, board_size=4, move_prob: list[float] = None, game_num=1, swap_player_for_each_game=True):
    """
    対局を開始する.

    Parameters
    ----------
    player_0: IPlayer
        1番目のプレイヤー(1回戦黒番).
    player_1: IPlayer
        2番目のプレイヤー(2回戦白番).
    board_size: int
        盤面のサイズ. 4以上8以下.
    move_prob: list[float]
        各マス目の着手成功確率. Noneの場合は乱数で初期化.
    game_num: int
        対局数.
    swap_player_for_each_game: bool
        1局ごとに先手後手を入れ替えるか.
    """
    if move_prob is None:
        move_prob = [random.random() for _ in range(board_size ** 2)]
    elif len(move_prob) != board_size ** 2:
        raise ValueError("Invalid length of move_prob.")

    players = (PlayerData(player=player_0, stats=PlayerStats()), PlayerData(player=player_1, stats=PlayerStats()))
    black_player, white_player = players

    for game_id in range(game_num):
        print(f"game {game_id + 1}:")

        __play_one_game(Position(board_size, move_prob), black_player, white_player)

        if swap_player_for_each_game:
            black_player, white_player = white_player, black_player

        for i, player in enumerate(players):
            stats = player.stats
            w, d, l, wr = stats.total_win_count, stats.total_draw_count, stats.total_loss_count, stats.total_win_rate
            print(f"{player.player.name}({i}): win-draw-loss (win_rate) = {w}-{d}-{l} ({wr * 100.0}%)")
        print()

def __play_one_game(pos: Position, black_player: PlayerData, white_player: PlayerData):
    player, opponent = black_player.player, white_player.player
    player.set_position(pos.copy())
    opponent.set_position(pos.copy())

    last_move = None
    succeeded_last_move = True
    while not pos.is_gameover():
        print(pos)
        if last_move is not None:
            s = "Success" if succeeded_last_move else "Failure"
            print(f"last_move = {pos.convert_coord_to_str(last_move)}({s})")
        print(f"player = {player.name}({pos.side_to_move.name})\n")

        if pos.can_pass():
            pos.do_pass()
            player.do_pass()
            opponent.do_pass()
            last_move = pos.PASS_COORD
            succeeded_last_move = True
        else:
            move_coord = player.gen_move()
            if not (move_coord in pos.get_next_moves()):
                print(f"Error: Player played invalid move at {pos.convert_coord_to_str(move_coord)}.")
                print("Suspend current game.")
                return

            move = pos.get_move(move_coord)
            pos.do_move(move)
            last_move = move.coord
            succeeded_last_move = (move.player == Player.CURRENT)

            player.do_move(move)
            opponent.do_move(move)

        player, opponent = opponent, player

    print("gameover")
    print(f"{pos}\n")

    if pos.side_to_move != DiscColor.BLACK:
        pos.do_pass()

    score = pos.get_score()
    if score == 0:
        black_player.stats.draw_count[DiscColor.BLACK] += 1
        white_player.stats.draw_count[DiscColor.WHITE] += 1
    elif score > 0:
        black_player.stats.win_count[DiscColor.BLACK] += 1
        white_player.stats.loss_count[DiscColor.WHITE] += 1
    else:
        black_player.stats.loss_count[DiscColor.BLACK] += 1
        white_player.stats.win_count[DiscColor.WHITE] += 1