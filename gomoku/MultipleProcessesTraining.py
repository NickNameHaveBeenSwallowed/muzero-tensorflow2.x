from convolution_model import model
from self_play import play_game
from trainer import Trainer

import multiprocessing
import threading
import datetime
import time

NUM_CHESS = 8
RENDER_BLOCK_SIZE = 50
OBSERVATION_SHAPE = (NUM_CHESS, NUM_CHESS, 3)
HIDDEN_STATE_CHANNEL = 32
NUM_SIMULATIONS = 400

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128

NUM_WORKERS = 8

def self_play_worker(pipe):
    worker_model = model(OBSERVATION_SHAPE, HIDDEN_STATE_CHANNEL, NUM_CHESS)
    while True:
        weights = pipe.recv()
        worker_model.representation.model.set_weights(weights[0])
        worker_model.dynamics.model.set_weights(weights[1])
        worker_model.prediction.model.set_weights(weights[2])

        self_play = play_game(NUM_CHESS, RENDER_BLOCK_SIZE, worker_model, NUM_SIMULATIONS, render=False)
        trajectory, winner = self_play.run()

        win = 1.0 if winner is not None else 0.0
        for i in trajectory[::-1]:
            i.append(win)
            win *= -1
        pipe.send(trajectory)

def save_model():
    global global_model
    while True:
        time.sleep(60)
        global_model.save_weights("./model/gomoku_{}X{}".format(NUM_CHESS, NUM_CHESS))
        print('\n save model at {}'.format(datetime.datetime.now()))

def training(trainer):
    global episode
    global global_model
    while True:
        policy_loss, value_loss, entropy = trainer.run_train(BATCH_SIZE, global_model)
        print("\r episode: {}, policy_loss: {}, value_loss: {}, losses: {}, entropy: {}, num_trajectory: {}".format(
            episode, policy_loss, value_loss, policy_loss + value_loss, entropy, len(trainer.replay_buffer.memory)),
        end="")

def communication(trainer, pipe_dict):
    global episode
    global global_model
    while True:
        for pipe in pipe_dict.values():
            pipe[0].send(
                [
                    global_model.representation.model.get_weights(),
                    global_model.dynamics.model.get_weights(),
                    global_model.prediction.model.get_weights()
                ]
            )

        for pipe in pipe_dict.values():
            trajectory = pipe[0].recv()
            trainer.replay_buffer.save_memory(trajectory)
        episode += 1

if __name__ == '__main__':
    global_model = model(OBSERVATION_SHAPE, HIDDEN_STATE_CHANNEL, NUM_CHESS)
    trainer = Trainer()

    episode = 0

    # global_model.load_weights("./model/gomoku_{}X{}".format(NUM_CHESS, NUM_CHESS))
    train_thread = threading.Thread(target=training, args=[trainer])
    train_thread.start()

    pipe_dict = {}
    for w in range(NUM_WORKERS):
        pipe_dict["worker_{}".format(str(w))] = multiprocessing.Pipe()

    process = []
    for w in range(NUM_WORKERS):
        self_play_process = multiprocessing.Process(
            target=self_play_worker,
            args=(
                pipe_dict["worker_{}".format(str(w))][1],
            )
        )
        process.append(self_play_process)
    [p.start() for p in process]

    communication_thread = threading.Thread(target=communication, args=[trainer, pipe_dict])
    communication_thread.start()

    savemodel_thread = threading.Thread(target=save_model)
    savemodel_thread.start()
