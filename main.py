from game import Game, HumanPlayer
from mc_player import MCPlayer
from random_mover import RandomMover
from dqn_player import DQNPlayer

player_0, player_1 = RandomMover(), DQNPlayer("qnet_31.h5")
# prob = [0.8, 0.2, 0.5, 0.5, 0.2, 0.8,
#         0.2, 0.2, 0.5, 1.0, 0.2, 0.2,
#         0.5, 0.5, 1.0, 1.0, 1.0, 0.5,
#         0.5, 1.0, 1.0, 1.0, 0.5, 0.5,
#         0.8, 0.8, 1.0, 0.5, 0.8, 0.8,
#         0.2, 0.8, 0.5, 0.5, 0.8, 0.2]
# game = Game(player_0, player_1, 6, prob)
game = Game(player_0, player_1, 6, [1.0] * 36)
game.start(200, swap_player_for_each_game=True, use_gui=False, gui_size=512)




