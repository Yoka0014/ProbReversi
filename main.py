from game import Game, HumanPlayer
from mc_player import MCPlayer
from mcts_player import MCTSPlayer
from pv_mcts_player import PV_MCTSPlayer
from random_mover import RandomMover
from dqn_player import DQNPlayer
from dualnet_player import DualNetPlayer

import mcts
import pv_mcts

#player_0, player_1 = RandomMover(), DQNPlayer("qnet_786.h5")
#player_0, player_1 = MCPlayer(100), RandomMover()
#player_0, player_1 = DualNetPlayer(6, "model_6x6_9b.h5"), MCPlayer(100)

config = pv_mcts.UCTConfig()
config.model_path = "model_6x6_9b.h5"
config.batch_size = 8
player_0, player_1 = PV_MCTSPlayer(config, 800), HumanPlayer()


BOARD_SIZE = 6
t=5
prob = []
for coord in range(BOARD_SIZE ** 2):
    prob.append(1.0 - t * 0.01 * (coord % (BOARD_SIZE + 1) + 3))

        
game = Game(player_0, player_1, BOARD_SIZE, prob)
game.start(400, swap_player_for_each_game=True, use_gui=False, gui_size=512)




