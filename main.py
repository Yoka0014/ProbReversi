import game
from random_mover import RandomMover

# このメソッドにプレイヤーを与えると対局できる.
player_0, player_1 = RandomMover(), RandomMover()
game.start(player_0, player_1, 4, None, 10)