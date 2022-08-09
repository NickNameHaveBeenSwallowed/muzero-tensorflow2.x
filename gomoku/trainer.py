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
        data_augmentation = self.data_augmentation(trajectory)
        for t in data_augmentation:
            self.memory.append(t)

    def sample(self, sample_size):
        batch_size = min(sample_size, len(self.memory))
        return random.sample(self.memory, batch_size)

    @staticmethod
    def data_augmentation(trajectory):
        t1, t2, t3, t4, t5, t6, t7, t8 = [], [], [], [], [], [], [], []
        for s_a_p_w in trajectory:
            state, action, policy, winner = s_a_p_w
            policy = np.reshape(policy, newshape=(action.shape[0], action.shape[1]))
            state_flip_1, action_flip_1, policy_flip_1, winner_flip_1 = tf.image.flip_left_right(state), tf.image.flip_left_right(action), tf.image.flip_left_right(policy), winner
            state_rot90, action_rot90, policy_rot90, winner_rot90 = tf.image.rot90(state, k=1), tf.image.rot90(action, k=1), tf.image.rot90(policy, k=1), winner
            state_flip_2, action_flip_2, policy_flip_2, winner_flip_2 = tf.image.flip_left_right(state_rot90), tf.image.flip_left_right(action_rot90), tf.image.flip_left_right(policy_rot90), winner
            state_rot180, action_rot180, policy_rot180, winner_rot180 = tf.image.rot90(state, k=2), tf.image.rot90(action, k=2), tf.image.rot90(policy, k=2), winner
            state_flip_3, action_flip_3, policy_flip_3, winner_flip_3 = tf.image.flip_left_right(state_rot180), tf.image.flip_left_right(action_rot180), tf.image.flip_left_right(policy_rot180), winner
            state_rot270, action_rot270, policy_rot270, winner_rot270 = tf.image.rot90(state, k=3), tf.image.rot90(action, k=3), tf.image.rot90(policy, k=3), winner
            state_flip_4, action_flip_4, policy_flip_4, winner_flip_4 = tf.image.flip_left_right(state_rot270), tf.image.flip_left_right(action_rot270), tf.image.flip_left_right(policy_rot270), winner
            t1.append([state, action, np.reshape(policy, newshape=(policy.shape[0] * policy.shape[1])), winner])
            t2.append([state_flip_1, action_flip_1, np.reshape(policy_flip_1, newshape=(policy_flip_1.shape[0] * policy_flip_1.shape[1])), winner_flip_1])
            t3.append([state_rot90, action_rot90, np.reshape(policy_rot90, newshape=(policy_rot90.shape[0] * policy_rot90.shape[1])), winner_rot90])
            t4.append([state_flip_2, action_flip_2, np.reshape(policy_flip_2, newshape=(policy_flip_2.shape[0] * policy_flip_2[1])), winner_flip_2])
            t5.append([state_rot180, action_rot180, np.reshape(policy_rot180, newshape=(policy_rot180[0] * policy_rot180[1])), winner_rot180])
            t6.append([state_flip_3, action_flip_3, np.reshape(policy_flip_3, newshape=(policy_flip_3.shape[0] * policy_flip_3.shape[1])), winner_flip_3])
            t7.append([state_rot270, action_rot270, np.reshape(policy_rot270, newshape=(policy_rot270.shape[0] * policy_rot270.shape[1])), winner_rot270])
            t8.append([state_flip_4, action_flip_4, np.reshape(policy_flip_4, newshape=(policy_flip_4.shape[0] * policy_flip_4.shape[1])), winner_flip_4])
        return t1, t2, t3, t4, t5, t6, t7, t8


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
