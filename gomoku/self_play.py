from game import Gomoku
from MCTS import MCTS_Chess
import numpy as np
import time

class play_game:
    def __init__(self, num_chess, block_size, model, num_simulations, render):
        self.num_chess = num_chess
        self.env = Gomoku(num_chess, block_size)
        self.render = render
        self.max_step = num_chess ** 2
        self.valid_action = list(range(num_chess ** 2))
        self.model = model
        self.mcts = MCTS_Chess
        self.num_simulations = num_simulations

    def choice_action(self, observation, T=1.0):
        # t = time.time()
        mcts = self.mcts(self.model, observation)
        visit_count = mcts.simulations(self.num_simulations)
        # print(visit_count.values())
        for k, v in visit_count.items():
            if k not in self.valid_action:
                visit_count[k] = 0

        action_visits = np.array(list(visit_count.values()))
        if np.any(action_visits):
            policy = action_visits ** (1 / T) / np.sum(action_visits ** (1 / T))
        else:
            policy = np.array([1 / len(self.valid_action) if i in self.valid_action else 0 for i in range(self.num_chess ** 2)])

        action = np.random.choice(len(policy), p=policy)
        self.valid_action.remove(action)
        # print(time.time() - t)
        return action, policy

    def run(self):
        trajectory = []
        state, winner = self.env.reset()
        # state = np.reshape(state, newshape=(self.num_chess, self.num_chess, 3))
        state = np.transpose(state, (1, 2, 0))
        if self.render:
            self.env.render()
        for step in range(self.max_step):
            action, policy = self.choice_action(state)
            action_onehot = np.reshape([1 if i == action else 0 for i in range(self.num_chess ** 2)], newshape=(self.num_chess, self.num_chess, 1))
            trajectory.append([state, action_onehot, policy])
            state, winner = self.env.step(action)

            if self.render:
                self.env.render()
            # state = np.reshape(state, newshape=(self.num_chess, self.num_chess, 3))
            state = np.transpose(state, (1, 2, 0))
            if winner is not None:
                break

        return trajectory, winner

# class human_play:
#     def __init__(self, num_chess, block_size, render):
#         self.num_chess = num_chess
#         self.env = Gomoku(num_chess, block_size)
#         self.render = render
#         self.max_step = num_chess ** 2
#
#     def run(self):
#         trajectory = []
#         state, winner = self.env.reset()
#         state = np.reshape(state, newshape=(self.num_chess, self.num_chess, 3))
#         if self.render:
#             self.env.render()
#         for step in range(self.max_step):
#             action = int(input())
#             policy = [1 if i == action else 0 for i in range(self.num_chess ** 2)]
#             action_onehot = np.reshape(policy, newshape=(self.num_chess, self.num_chess, 1))
#             policy = np.array(policy)
#             trajectory.append([state, action_onehot, policy])
#             state, winner = self.env.step(action)
#             if self.render:
#                 self.env.render()
#             state = np.reshape(state, newshape=(self.num_chess, self.num_chess, 3))
#             if winner is not None:
#                 break
#         last_action = np.reshape([0 for _ in range(self.num_chess ** 2)], newshape=(self.num_chess, self.num_chess, 1))
#         last_policy = np.array([0 for _ in range(self.num_chess ** 2)])
#         trajectory.append([state, last_action, last_policy])
#         return trajectory, winner