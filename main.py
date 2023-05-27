from game import Game, HumanPlayer
from mc_player import MCPlayer
from random_mover import RandomMover

player_0, player_1 = HumanPlayer(), MCPlayer()
prob = [0.8, 0.2, 0.5, 0.5, 0.2, 0.8,
        0.2, 0.2, 0.5, 1.0, 0.2, 0.2,
        0.5, 0.5, 1.0, 1.0, 1.0, 0.5,
        0.5, 1.0, 1.0, 1.0, 0.5, 0.5,
        0.8, 0.8, 1.0, 0.5, 0.8, 0.8,
        0.2, 0.8, 0.5, 0.5, 0.8, 0.2]
game = Game(player_0, player_1, 6, prob)
game.start(1, swap_player_for_each_game=False, use_gui=True, gui_size=512)



