from resnet_model import model
from self_play import play_game

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NUM_CHESS = 9
RENDER_BLOCK_SIZE = 50
OBSERVATION_SHAPE = (NUM_CHESS, NUM_CHESS, 3)
HIDDEN_STATE_CHANNEL = 32
NUM_SIMULATIONS = 400

def self_play(model, num_simulations):
    self_play = play_game(NUM_CHESS, RENDER_BLOCK_SIZE, model, num_simulations, True)
    trajectory, winner = self_play.run()
    print(winner)

if __name__ == '__main__':
    gomoku_model = model(OBSERVATION_SHAPE, HIDDEN_STATE_CHANNEL, NUM_CHESS)
    gomoku_model.load_weights("./model/gomoku_{}X{}".format(NUM_CHESS, NUM_CHESS))
    self_play(gomoku_model, NUM_SIMULATIONS)
