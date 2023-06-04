from game import Game, HumanPlayer
from mc_player import MCPlayer
from random_mover import RandomMover
from dqn_player import DQNPlayer

#player_0, player_1 = RandomMover(), DQNPlayer("qnet_786.h5")
player_0, player_1 = DQNPlayer("qnet_153.h5"), HumanPlayer()
#player_0, player_1 = RandomMover(), DQNPlayer("qnet_prob0_final.h5")
#player_0, player_1 = HumanPlayer(), MCPlayer(100)
#player_0, player_1 = MCPlayer(100), MCPlayer(100)


size = 6
prob = [1.0] * size ** 2

        
game = Game(player_0, player_1, size, prob)
game.start(1000, swap_player_for_each_game=True, use_gui=False, gui_size=512)




