from game import Game
from random_mover import RandomMover

# このメソッドにプレイヤーを与えると対局できる.
player_0, player_1 = RandomMover(), RandomMover()
game = Game(player_0, player_1, 6)
game.start(10, swap_player_for_each_game=True, use_gui=True)