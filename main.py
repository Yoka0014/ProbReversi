from game import Game, HumanPlayer
from random_mover import RandomMover
from mc_player import MCPlayer

# このメソッドにプレイヤーを与えると対局できる.
player_0, player_1 = HumanPlayer(), MCPlayer()
game = Game(player_0, player_1, 8)
game.start(10, swap_player_for_each_game=True, use_gui=False, gui_size=512)

