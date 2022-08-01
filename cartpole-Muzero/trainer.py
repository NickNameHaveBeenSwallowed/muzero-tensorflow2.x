from tensorflow.keras import optimizers, losses
from collections import deque
import tensorflow as tf
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, max_size, unroll_steps):
        self.memory = deque(maxlen=max_size)
        self.unroll = deque(maxlen=unroll_steps)
        self.unroll_steps = unroll_steps

    def save_memory(self, state, policy, action, reward, value_next, next_state, done):
        self.unroll.append([state, policy, action, reward, value_next, next_state, done])
        if len(self.unroll) == self.unroll_steps:
            self.memory.append(list(self.unroll))

    def simple(self, simple_size):
        batchs = min(simple_size, len(self.memory))
        return random.sample(self.memory, batchs)

def process_data(data):
    data = np.array(data)
    new_data = []
    for d in data:
        new_data.append(d.T)
    new_data = np.array(new_data).T

    return new_data

class Trainer:
    def __init__(self, discount, lr=2e-3):
        self.optimizer = optimizers.Adam(lr)
        self.discount = discount

    def update_weight(self, model, replay_buffer, batch_size):
        data = replay_buffer.simple(batch_size)
        data = process_data(data)
        with tf.GradientTape() as tape:
            first_observations = np.array(list(data[0][0]))

            hidden_state = model.representation.model(first_observations)
            policy_targets, value_targets, reward_targets = [], [], []
            policy_preds, value_preds, reward_preds = [], [], []
            for step in range(len(data)):
                policy_pred, value_pred = model.prediction.model(hidden_state)
                action = np.array(list(data[step][2]))

                hidden_state, reward_pred = model.dynamics.model([hidden_state, action])
                policy_target = np.array(list(data[step][1]))
                value_target = data[step][3] + self.discount * (1 - data[step][6]) * data[step][4]
                value_target = np.reshape(value_target, newshape=(-1, 1))

                policy_targets.append(policy_target)
                value_targets.append(value_target)
                reward_targets.append(np.reshape(data[step][3], newshape=(-1, 1)))

                policy_preds.append(policy_pred)
                value_preds.append(value_pred)
                reward_preds.append(reward_pred)

            entropys = []
            for policy in policy_preds:
                entropy = - np.sum(policy[0] * np.log(policy[0] + 1e-8))
                entropys.append(entropy)

            policy_loss = losses.categorical_crossentropy(
                y_pred=policy_preds,
                y_true=policy_targets
            )
            value_loss = losses.mean_squared_error(
                y_pred=value_preds,
                y_true=value_targets
            )
            reward_loss = losses.mean_squared_error(
                y_pred=reward_preds,
                y_true=reward_targets
            )

            loss = policy_loss + value_loss + reward_loss
            grad = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return tf.reduce_mean(policy_loss), tf.reduce_mean(value_loss), tf.reduce_mean(reward_loss), tf.reduce_mean(entropys)
