from game import Game, HumanPlayer
from mc_player import MCPlayer
from mcts_player import MCTSPlayer
from random_mover import RandomMover
from dqn_player import DQNPlayer

import mcts

#player_0, player_1 = RandomMover(), DQNPlayer("qnet_786.h5")
#player_0, player_1 = MCPlayer(100), RandomMover()
player_0, player_1 = MCTSPlayer(mcts.UCTConfig(), 10000), HumanPlayer()
#player_0, player_1 = MCPlayer(100), MCPlayer(100)


BOARD_SIZE = 8
t = 5
prob = [1.0] * 64
# prob = []
# for coord in range(BOARD_SIZE ** 2):
#     prob.append(1.0 - t * 0.01 * (coord % (BOARD_SIZE + 1) + 3))

        
game = Game(player_0, player_1, BOARD_SIZE, prob)
game.start(1, swap_player_for_each_game=True, use_gui=False, gui_size=512)




