from game import TicTacToe
from MCTS import MCTS_Chess
import numpy as np

class play_game:
    def __init__(self, block_size, model, num_simulations, render):
        self.env = TicTacToe(block_size)
        self.render = render
        self.valid_action = list(range(9))
        self.model = model
        self.mcts = MCTS_Chess
        self.num_simulations = num_simulations

    def choice_action(self, observation, T=1.0, add_noise=True):
        mcts = self.mcts(self.model, observation)
        visit_count = mcts.simulations(self.num_simulations, add_noise=add_noise)
        for k, v in visit_count.items():
            if k not in self.valid_action:
                visit_count[k] = 0

        action_visits = np.array(list(visit_count.values()))
        if np.any(action_visits):
            policy = action_visits ** (1 / T) / np.sum(action_visits ** (1 / T))
        else:
            policy = np.array([1 / len(self.valid_action) if i in self.valid_action else 0 for i in range(9)])
        # print(policy, "\t\t", - np.sum(policy * np.log(policy + 1e-8)))
        action = np.random.choice(len(policy), p=policy)
        self.valid_action.remove(action)
        return action, policy

    def run(self, T=1.0, add_noise=True):
        trajectory = []
        state, winner = self.env.reset()
        state = np.transpose(state, (1, 2, 0))
        if self.render:
            self.env.render()
        for step in range(9):
            action, policy = self.choice_action(state, T, add_noise)
            action_onehot = np.reshape([1 if i == action else 0 for i in range(9)], newshape=(3, 3, 1))
            trajectory.append([state, action_onehot, policy])
            state, winner = self.env.step(action)

            if self.render:
                self.env.render()
            state = np.transpose(state, (1, 2, 0))
            if winner is not None:
                break

        return trajectory, winner
