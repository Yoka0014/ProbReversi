from game import Game
from myplayer import MyPlayer

BOARD_SIZE = 6  # 盤面サイズ(大会は6と8で行う)
NUM_GAME = 10   # 対戦回数

t = 5
prob = []
for coord in range(BOARD_SIZE ** 2):
    prob.append(1.0 - t * 0.01 * (coord % (BOARD_SIZE + 1) + 3))


player_0, player_1 = MyPlayer(), MyPlayer()     # ここに対戦するプレイヤーを設定

# 先手後手入れ替えながらNUM_GAME回対戦
game = Game(player_0, player_1, BOARD_SIZE, prob)
game.start(NUM_GAME, swap_player_for_each_game=True, use_gui=True, gui_size=512)




