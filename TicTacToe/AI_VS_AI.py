from resnet_model import model
from self_play import play_game

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RENDER_BLOCK_SIZE = 50
HIDDEN_STATE_CHANNEL = 32
NUM_SIMULATIONS = 30

def self_play(model, num_simulations):
    self_play = play_game(RENDER_BLOCK_SIZE, model, num_simulations, True)
    trajectory, winner = self_play.run(add_noise=False)
    print(winner)

if __name__ == '__main__':
    tictactoe_model = model((3, 3, 3), HIDDEN_STATE_CHANNEL, 3)
    tictactoe_model.load_weights("./model/tictactoe_")
    self_play(tictactoe_model, NUM_SIMULATIONS)
