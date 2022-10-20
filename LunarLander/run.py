from network import linner_model
from trainer import ReplayBuffer, Trainer
from MCTS import MCTS_Atari
import matplotlib.pyplot as plt

import numpy as np
import copy
import gym

ENV_NAME = "LunarLander-v2"
OBSERVATION_SIZE = 8
ACTION_SIZE = 4

DISCOUNT = 0.997
NUM_SIMULATIONS = 50
UNROLL_STEPS = 6
MEMORY_SIZE = int(1e6)
SIMPLE_SIZE = 1024
EPISODES = 300

class muzero:
    def __init__(self, observation_size, action_size):
        self.model = linner_model(observation_size, action_size)
        self.MCTS = MCTS_Atari

    def choice_action(self, observation, T=1.0):
        MCTS = self.MCTS(self.model, observation)
        visit_count, MCTS_value = MCTS.simulations(NUM_SIMULATIONS, DISCOUNT)
        visit_counts = list(visit_count.values())
        prob = np.array(visit_counts) ** (1 / T) / np.sum(np.array(visit_counts) ** (1 / T))
        return np.random.choice(len(prob), p=prob), prob, MCTS_value

    def plot_score(self, scores, avg_scores):
        plt.plot(scores)
        plt.plot(avg_scores)
        plt.show()

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    agent = muzero(OBSERVATION_SIZE, ACTION_SIZE)
    agent.model.load_weights("./LunarLander")
    trainer = Trainer(discount=DISCOUNT)
    replay_buffer = ReplayBuffer(MEMORY_SIZE, UNROLL_STEPS)

    scores = []
    avg_scores = []
    for e in range(EPISODES):

        state = env.reset()
        action_next, policy_next, _ = agent.choice_action(state)
        rewards = 0
        while True:
            # env.render()
            action, policy = action_next, policy_next
            next_state, reward, done, _ = env.step(action)
            action_next, policy_next, value_next = agent.choice_action(next_state)
            done = 1 if done else 0

            rewards += reward
            action_onehot = np.array([1 if i == action else 0 for i in range(ACTION_SIZE)])
            replay_buffer.save_memory(state, policy, action_onehot, reward, value_next, next_state, done)
            state = copy.deepcopy(next_state)

            if done: break

        scores.append(rewards)
        avg_scores.append(sum(scores)/len(scores))
        policy_loss, value_loss, reward_loss, policy_entropy = trainer.update_weight(agent.model, replay_buffer, SIMPLE_SIZE)

        print("episode: {}/{}, policy_loss: {}, value_loss: {}, reward_loss: {}, policy_entropy: {}, score: {}".format(
            e + 1, EPISODES, policy_loss, value_loss, reward_loss, policy_entropy, rewards))
    agent.model.save_weights("./LunarLander")
    agent.plot_score(scores, avg_scores)
