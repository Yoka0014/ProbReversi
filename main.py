from game import Game, HumanPlayer
from random_mover import RandomMover

# このメソッドにプレイヤーを与えると対局できる.
player_0, player_1 = HumanPlayer(), RandomMover()
game = Game(player_0, player_1, 4)
game.start(1, swap_player_for_each_game=True, use_gui=True, gui_size=512)