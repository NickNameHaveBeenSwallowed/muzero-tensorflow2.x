from gym.envs.classic_control import rendering
import numpy as np
import gym

def check(filter, state, size, filter_w, filter_h):
    done = False
    result = []
    for i in range(size - filter_h + 1):
        for j in range(size - filter_w + 1):
            input_block = state[i:i + filter_h, j:j + filter_w]
            result.append(np.sum(filter * input_block))

    for i in result:
        if i == 5:
            done =True
    return done

class Gomoku(gym.Env):
    def __init__(self, num_chess, block_size):

        if num_chess < 5:
            raise ValueError("The minimum checkerboard is 5.")

        self.board = None
        self.num_chess = num_chess
        self.winner = None

        self.block_size = block_size

        self.viewer = None

        self.player = None

    def reset(self):
        self.board = np.zeros([3, self.num_chess, self.num_chess])
        self.player = 0

        return self.board, self.winner

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(
            self.num_chess * self.block_size,
            self.num_chess * self.block_size
            )
            self.viewer.geoms.clear()
            self.viewer.onetime_geoms.clear()
        for i in range(self.num_chess - 1):
            line = rendering.Line((0, (i+1) * self.block_size), (self.num_chess * self.block_size, (i+1) * self.block_size))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)
            line = rendering.Line(((i+1) * self.block_size, 0), ((i+1) * self.block_size, self.num_chess * self.block_size))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

        for i in range(self.num_chess):
            for j in range(self.num_chess):
                if self.board[0][j][i] == 1:
                    circle = rendering.make_circle(0.35 * self.block_size)
                    circle.set_color(0 / 255, 139 / 255, 0 / 255)
                    move = rendering.Transform(
                        translation=(
                            (i + 0.5) * self.block_size,
                            (self.num_chess - j - 0.5) * self.block_size
                        )
                    )
                    circle.add_attr(move)
                    self.viewer.add_geom(circle)

        for i in range(self.num_chess):
            for j in range(self.num_chess):
                if self.board[1][j][i] == 1:
                    circle = rendering.make_circle(0.35 * self.block_size)
                    circle.set_color(238 / 255,  118 / 255, 33 / 255)
                    move = rendering.Transform(
                        translation=(
                            (i + 0.5) * self.block_size,
                            (self.num_chess - j - 0.5) * self.block_size
                        )
                    )
                    circle.add_attr(move)
                    self.viewer.add_geom(circle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def done(self):
        done = False
        filter0 = np.array([1, 1, 1, 1, 1])
        filter1 = np.array([[1], [1], [1], [1], [1]])
        filter2 = np.eye(5)
        filter3 = np.eye(5)[::-1]
        done = check(filter0, self.board[0], self.num_chess, 5, 1) or done
        done = check(filter0, self.board[1], self.num_chess, 5, 1) or done
        done = check(filter1, self.board[0], self.num_chess, 1, 5) or done
        done = check(filter1, self.board[1], self.num_chess, 1, 5) or done
        done = check(filter2, self.board[0], self.num_chess, 5, 5) or done
        done = check(filter2, self.board[1], self.num_chess, 5, 5) or done
        done = check(filter3, self.board[0], self.num_chess, 5, 5) or done
        done = check(filter3, self.board[1], self.num_chess, 5, 5) or done
        return done

    def step(self, action: int):
        i = int(action / self.num_chess)
        j = action % self.num_chess
        if self.board[0][i][j] == 1 or self.board[1][i][j] == 1:
            raise ValueError("Action error, there are pieces here")
        else:
            self.board[self.player][i][j] = 1

        if self.done():
            self.winner = self.player
            if self.player == 0:
                self.board[2] = np.ones([self.num_chess, self.num_chess])
                self.player = 1
            else:
                self.board[2] = np.zeros([self.num_chess, self.num_chess])
                self.player = 0
            return self.board, self.winner

        else:
            if self.player == 0:
                self.board[2] = np.ones([self.num_chess, self.num_chess])
                self.player = 1
            else:
                self.board[2] = np.zeros([self.num_chess, self.num_chess])
                self.player = 0
            return self.board, self.winner