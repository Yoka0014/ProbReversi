from game import Game, HumanPlayer
from mc_player import MCPlayer
from random_mover import RandomMover
from dqn_player import DQNPlayer

player_0, player_1 = MCPlayer(), DQNPlayer("qnet_624.h5")
# prob = [0.8, 0.2, 0.5, 0.5, 0.2, 0.8,
#         0.2, 0.2, 0.5, 1.0, 0.2, 0.2,
#         0.5, 0.5, 1.0, 1.0, 1.0, 0.5,
#         0.5, 1.0, 1.0, 1.0, 0.5, 0.5,
#         0.8, 0.8, 1.0, 0.5, 0.8, 0.8,
#         0.2, 0.8, 0.5, 0.5, 0.8, 0.2]
# game = Game(player_0, player_1, 6, prob)

def beep(freq, dur=100):
    """
        ビープ音を鳴らす.
        @param freq 周波数
        @param dur  継続時間（ms）
    """
    import platform
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(freq, dur)
    else:
        import os
        os.system('play -n synth %s sin %s' % (dur/1000, freq))


game = Game(player_0, player_1, 6, [1.0] * 36)
game.start(1000, swap_player_for_each_game=True, use_gui=False, gui_size=512)

beep(2000, 1000)




