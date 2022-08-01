import numpy as np

PB_C_INIT = 1.25
PB_C_BASE = 19652

class MinMax:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class TreeNode:
    def __init__(self):
        self.parent = None
        self.prior = 1.0
        self.hidden_state = None
        self.children = {}
        self.visit_count = 0
        self.reward = 0
        self.Q = 0

    def is_leaf_Node(self):
        return self.children == {}

    def is_root_Node(self):
        return self.parent is None

def add_exploration_noise(node, dirichlet_alpha=0.3, exploration_fraction=0.25):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    frac = exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

def ucb_score_Atari(node, minmax, pb_c_init=PB_C_INIT, pb_c_base=PB_C_BASE):
    pb_c = np.log(
        (node.parent.visit_count + pb_c_base + 1) / pb_c_base
    ) + pb_c_init
    pb_c *= np.sqrt(node.parent.visit_count) / (node.visit_count + 1)
    prior_score = pb_c * node.prior
    return minmax.normalize(node.Q) + prior_score

def select_argmax_pUCB_child_Atari(node, minmax):
    return max(
        node.children.items(),
        key=lambda key_node_tuple: ucb_score_Atari(key_node_tuple[1], minmax)
    )

def expand_Atari(node, model):
    if node.parent is not None:
        node.hidden_state, node.reward = model.dynamics.predict(node.parent.hidden_state, node.action)
    policy, value = model.prediction.predict(node.hidden_state)
    node.Q = value
    keys = list(range(len(policy)))
    for k in keys:
        child = TreeNode()
        child.action = k
        child.prior = policy[k]
        child.parent = node
        node.children[k] = child

def backpropagate_Atari(node, minmax, discount):
    value = node.Q
    while True:
        node.visit_count += 1
        minmax.update(node.Q)
        if node.is_root_Node():
            break
        else:
            value = node.reward + discount * value
            node = node.parent
            node.Q = (node.Q * node.visit_count + value) / (node.visit_count + 1)

class MCTS_Atari:
    def __init__(self, model, observation):
        self.root_Node = TreeNode()
        self.model = model
        self.root_Node.hidden_state = self.model.representation.predict(observation)
        self.minmax = MinMax()

    def simulations(self, num_simulation, discount, add_noise=True):
        for _ in range(num_simulation + 1):
            node = self.root_Node
            while True:
                if node.is_leaf_Node():break
                else:
                    _key, node = select_argmax_pUCB_child_Atari(node, self.minmax)
            expand_Atari(node, self.model)
            if node == self.root_Node and add_noise:
                add_exploration_noise(node)
            backpropagate_Atari(node, self.minmax, discount)
        action_visits = {}
        for k, n in self.root_Node.children.items():
            action_visits[k] = n.visit_count
        return action_visits, self.root_Node.Q

    def __str__(self):
        return "Muzero_MCTS_Atari"

def ucb_score_Chess(node, minmax, pb_c_init=PB_C_INIT, pb_c_base=PB_C_BASE):
    pb_c = np.log(
        (node.parent.visit_count + pb_c_base + 1) / pb_c_base
    ) + pb_c_init
    pb_c *= np.sqrt(node.parent.visit_count) / (node.visit_count + 1)
    prior_score = pb_c * node.prior
    return minmax.normalize(node.Q) + prior_score

def select_argmax_pUCB_child_Chess(node, minmax):
    return max(
        node.children.items(),
        key=lambda key_node_tuple: ucb_score_Chess(key_node_tuple[1], minmax)
    )

def expand_Chess(node, model):
    if node.parent is not None:
        node.hidden_state = model.dynamics.predict(node.parent.hidden_state, node.action)
    policy, value = model.prediction.predict(node.hidden_state)
    node.Q = value
    keys = list(range(len(policy)))
    for k in keys:
        child = TreeNode()
        child.action = k
        child.prior = policy[k]
        child.parent = node
        node.children[k] = child

def backpropagate_Chess(node, minmax):
    value = node.Q
    while True:
        node.visit_count += 1
        minmax.update(node.Q)
        if node.is_root_Node():
            break
        else:
            value = - value
            node = node.parent
            node.Q = (node.Q * node.visit_count + value) / (node.visit_count + 1)

class MCTS_Chess:
    def __init__(self, model, observation):
        self.root_Node = TreeNode()
        self.model = model
        self.root_Node.hidden_state = self.model.representation.predict(observation)
        self.minmax = MinMax()

    def simulations(self, num_simulation, add_noise=True):
        for _ in range(num_simulation + 1):
            node = self.root_Node
            while True:
                if node.is_leaf_Node(): break
                else:
                    key, node = select_argmax_pUCB_child_Chess(node, self.minmax)
            expand_Chess(node, self.model)
            if node == self.root_Node and add_noise:
                add_exploration_noise(node)
            backpropagate_Chess(node, self.minmax)

        action_visits = {}
        for k, n in self.root_Node.children.items():
            action_visits[k] = n.visit_count
        return action_visits

    def __str__(self):
        return "Muzero_MCTS_Chess"
