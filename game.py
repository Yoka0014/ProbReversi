"""
確率リバーシの対局を行うためのコード.
"""
from abc import ABCMeta, abstractproperty, abstractmethod  # 抽象クラスを定義するために必要.
from dataclasses import dataclass
from functools import partial
from typing import Callable
import os
import random
import time

from prob_reversi import DiscColor, Player, Move, Position
from gui import GameGUI


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


class HumanPlayer(IPlayer):
    def __init__(self):
        self.__pos = Position(4)
        self.human_input: int = None

    @property
    def name(self):
        return "Human"

    def set_position(self, pos: Position):
        self.__pos = pos

    def gen_move(self) -> int:
        if self.__pos.can_pass():
            return self.__pos.PASS_COORD

        while self.human_input is None or not self.__pos.is_leagal(self.human_input):
            time.sleep(0)   # ビジー状態にならないようにsleepを挟む.

        ret = self.human_input
        self.human_input = None
        return ret

    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()


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


class Game:
    def __init__(self, player_0: IPlayer, player_1: IPlayer, board_size=4, trans_prob: list[float] = None):
        """
        Gameオブジェクトを生成する.

        Parameters
        ----------
        player_0: IPlayer
            1番目のプレイヤー(1回戦黒番).
        player_1: IPlayer
            2番目のプレイヤー(1回戦白番).
        board_size: int
            盤面のサイズ. 4以上8以下.
        trans_prob: list[float]
            各マス目の着手成功確率. Noneの場合は乱数で初期化.
        """
        self.__players = [PlayerData(PlayerStats(), player_0), PlayerData(PlayerStats(), player_1)]

        if trans_prob is None:
            trans_prob = [random.random() for _ in range(board_size ** 2)]

        self.__pos = Position(board_size, trans_prob)
        self.__last_move: Move = None

        self.__show_pos: Callable[[Position], None]

        self.__gui: GameGUI = None

    def get_pos(self) -> Position:
        return self.__pos.copy()

    def start(self, game_num=1, swap_player_for_each_game=True, use_gui=True, gui_size=512):
        """
        対局を開始する.

        Parameters
        ----------
        game_num: int
            対局数.
        swap_player_for_each_game: bool
            1局ごとに先手後手を入れ替えるか.
        use_gui: bool
            GUIで盤面を表示するか.
        gui_size: int
            GUIのクライアント領域のサイズ.
        """
        if any(isinstance(p.player, HumanPlayer) for p in self.__players):
            use_gui = True

        if use_gui:
            worker = partial(self.__start, game_num=game_num, swap_player_for_each_game=swap_player_for_each_game)
            self.__gui = GameGUI(worker, gui_size)
            self.__show_pos = lambda pos: self.__gui.set_position(pos)
            self.__gui.window_closed = lambda: os._exit(0)  # ToDo: 対局中にウインドウを閉じられると, 対局スレッドだけが生き続けるので, 
                                                            # os._exitで強制終了させている. あまり良いやり方ではないので要修正.
            self.__gui.start()
        else:
            self.__gui = None
            self.__show_pos = lambda pos: print(f"{pos}\n")
            self.__start(game_num, swap_player_for_each_game)

    def __start(self, game_num, swap_player_for_each_game):
        black_player, white_player = self.__players

        for game_id in range(game_num):
            print(f"game {game_id + 1}:")
            print(f"{black_player.player.name} v.s. {white_player.player.name}")

            self.__play_one_game(black_player, white_player)

            if swap_player_for_each_game:
                black_player, white_player = white_player, black_player

            for i, player in enumerate(self.__players):
                stats = player.stats
                w, d, l, wr = stats.total_win_count, stats.total_draw_count, stats.total_loss_count, stats.total_win_rate
                print(f"{player.player.name}({i}): win-draw-loss (win_rate) = {w}-{d}-{l} ({wr * 100.0}%)")
            print()

    def __play_one_game(self, black_player: PlayerData, white_player: PlayerData):

        player, opponent = black_player.player, white_player.player

        pos = self.__pos
        pos.clear()

        player.set_position(pos.copy())
        opponent.set_position(pos.copy())

        while not pos.is_gameover():
            self.__show_pos(pos)
            if self.__last_move is not None:
                s = "Success" if self.__last_move.player == Player.CURRENT else "Failure"
                print(f"last_move = {pos.convert_coord_to_str(self.__last_move.coord)}({s})")
            print(f"player = {player.name}({pos.side_to_move.name})\n")

            if pos.can_pass():
                pos.do_pass()
                player.do_pass()
                opponent.do_pass()
                self.__last_move = Move(Player.CURRENT, pos.PASS_COORD)
            else:
                if self.__gui is not None:
                    if isinstance(player, HumanPlayer):
                        player.__class__ = HumanPlayer

                        def f(c):
                            player.human_input = c
                        self.__gui.board_clicked = f

                move_coord = player.gen_move()
                if self.__gui is not None and isinstance(player, HumanPlayer):
                    self.__gui.board_clicked = lambda c: None

                if not (move_coord in pos.get_next_moves()):
                    print(f"Error: Player played invalid move at {pos.convert_coord_to_str(move_coord)}.")
                    print("Suspend current game.")
                    return

                move = pos.get_move(move_coord)
                pos.do_move(move)
                self.__last_move = move

                player.do_move(move)
                opponent.do_move(move)

            player, opponent = opponent, player

        print("gameover")
        self.__show_pos(pos)

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
