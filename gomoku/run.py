from self_play import play_game
from convolution_model import model
from trainer import Trainer

NUM_CHESS = 8
RENDER_BLOCK_SIZE = 50
OBSERVATION_SHAPE = (NUM_CHESS, NUM_CHESS, 3)
HIDDEN_STATE_CHANNEL = 32
NUM_SIMULATIONS = 400
EPISODES = 10000

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128

if __name__ == '__main__':
    import time
    t = time.time()
    gomoku_model = model(OBSERVATION_SHAPE, HIDDEN_STATE_CHANNEL, NUM_CHESS)
    # gomoku_model.load_weights("./model/gomoku_{}X{}".format(NUM_CHESS, NUM_CHESS))
    trainer = Trainer()

    for e in range(EPISODES):
        self_play = play_game(NUM_CHESS, BATCH_SIZE, gomoku_model, NUM_SIMULATIONS, render=False)
        trajectory, winner = self_play.run()

        win = 1.0 if winner is not None else 0.0
        for i in trajectory[::-1]:
            i.append(win)
            win *= -1

        trainer.replay_buffer.save_memory(trajectory)

        ploss, vloss, ent = trainer.run_train(BUFFER_SIZE, gomoku_model)
        print("episode: {}/{}, policy_loss: {}, value_loss: {}, losses: {}, entropy: {}, memory_size: {}, run time: {}s".format(
            e + 1, EPISODES, ploss, vloss, ploss + vloss, ent, len(trainer.replay_buffer.memory), time.time() - t
        ))
        gomoku_model.save_weights("./model/gomoku_{}X{}".format(NUM_CHESS, NUM_CHESS))
