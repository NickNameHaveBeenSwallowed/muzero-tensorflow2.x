from self_play import play_game
from resnet_model import model
from trainer import Trainer

import matplotlib.pyplot as plt

RENDER_BLOCK_SIZE = 50
HIDDEN_STATE_CHANNEL = 32
NUM_SIMULATIONS = 30
EPISODES = 1000

BUFFER_SIZE = 3000
BATCH_SIZE = 64

if __name__ == '__main__':
    import time
    t = time.time()
    tictactoe_model = model((3, 3, 3), HIDDEN_STATE_CHANNEL, 3)
    tictactoe_model.load_weights("./model/tictactoe_")
    trainer = Trainer(max_save_memory=BUFFER_SIZE)

    losses, policy_loss, value_loss, entropys = [], [], [], []
    for e in range(EPISODES):
        self_play = play_game(RENDER_BLOCK_SIZE, tictactoe_model, NUM_SIMULATIONS, render=False)
        trajectory, winner = self_play.run()

        win = 1.0 if winner is not None else 0.0
        for i in trajectory[::-1]:
            i.append(win)
            win *= -1

        trainer.replay_buffer.save_memory(trajectory)

        ploss, vloss, ent = trainer.run_train(BATCH_SIZE, tictactoe_model)
        print("episode: {}/{}, policy_loss: {}, value_loss: {}, losses: {}, entropy: {}, memory_size: {}, run time: {}s".format(
            e + 1, EPISODES, ploss, vloss, ploss + vloss, ent, len(trainer.replay_buffer.memory), time.time() - t
        ))
        losses.append(ploss + vloss)
        policy_loss.append(ploss)
        value_loss.append(vloss)
        entropys.append(ent)
        tictactoe_model.save_weights("./model/tictactoe")

    fig, ax = plt.subplots()
    ax.plot(losses, label="loss")
    ax.plot(policy_loss, label="policy loss")
    ax.plot(value_loss, label="value loss")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(entropys)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Entropy')
    ax.legend()
    plt.show()
