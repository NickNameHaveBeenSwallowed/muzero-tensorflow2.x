from collections import deque

import numpy as np
from tensorflow.keras import optimizers, losses

import tensorflow as tf
import random

class ReplayBuffer():
    def __init__(self, max_memory):
        self.memory = deque(maxlen=max_memory)
        self.len = len(self.memory)

    def save_memory(self, trajectory):
        self.memory.append(trajectory)

    def sample(self, sample_size):
        batch_size = min(sample_size, len(self.memory))
        return random.sample(self.memory, batch_size)


class Trainer():
    def __init__(self, lr=2e-3, max_save_memory=int(1e6)):
        self.optimizer = optimizers.Adam(lr)
        self.replay_buffer = ReplayBuffer(max_save_memory)

    def run_train(self, batch_size, model):
        train_data = self.replay_buffer.sample(batch_size)
        policys_losses, value_losses, entropys = [], [], []
        for data in train_data:
            with tf.GradientTape() as tape:
                first_state = np.array([data[0][0]])
                hidden_state = model.representation.model(first_state)
                policy_targets, value_targets = [], []
                policy_predicts,  value_predicts = [], []
                for step in range(len(data)):
                    p_pred, v_pred = model.prediction.model(hidden_state)
                    act = np.array([data[step][1]])
                    hidden_state = model.dynamics.model([hidden_state, act])
                    p_tar = np.array([data[step][2]])
                    v_tar = np.reshape(data[step][3], newshape=(-1, 1))

                    policy_targets.append(p_tar)
                    value_targets.append(v_tar)

                    policy_predicts.append(p_pred)
                    value_predicts.append(v_pred)

                entropy = []
                for policy in policy_predicts:
                    entropy.append(- np.sum(policy[0] * np.log(policy[0] + 1e-8)))

                policy_loss = losses.categorical_crossentropy(
                    y_pred=policy_predicts,
                    y_true=policy_targets
                )
                value_loss = losses.mean_squared_error(
                    y_pred=value_predicts,
                    y_true=value_targets
                )

                loss = policy_loss + value_loss

            grad = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, model.trainable_variables))
            policys_losses.append(np.mean(policy_loss))
            value_losses.append(np.mean(value_loss))
            entropys.append(np.mean(entropy))

        return np.mean(policys_losses), np.mean(value_losses), np.mean(entropys)