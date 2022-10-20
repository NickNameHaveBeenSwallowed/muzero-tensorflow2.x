from tensorflow.keras import layers, Input, Model
import numpy as np

class representation:
    def __init__(self, observation_size):
        observation = Input(shape=(observation_size))
        x = layers.Flatten()(observation)
        x = layers.Dense(units=1024, activation='relu')(x)
        x = layers.Dense(units=1024, activation='relu')(x)
        hidden_state = layers.Dense(units=observation_size, activation='tanh')(x)
        self.model = Model(inputs=observation, outputs=hidden_state)
        # self.model.summary()
        self.trainable_variables = self.model.trainable_variables

    def predict(self, observation):
        observation = np.array([observation])
        hidden_state = np.array(self.model(observation)[0])
        return hidden_state

class dynamics:
    def __init__(self, observation_size, action_size):
        self.action_size = action_size
        hidden_state = Input(shape=(observation_size))
        action = Input(shape=(action_size))
        x = layers.concatenate([hidden_state, action])
        x = layers.Dense(units=1024, activation='relu')(x)
        x = layers.Dense(units=1024, activation='relu')(x)
        next_hidden_state = layers.Dense(units=observation_size, activation='tanh')(x)
        reward = layers.Dense(units=1)(next_hidden_state)
        self.model = Model(inputs=[hidden_state, action], outputs=[next_hidden_state, reward])
        # self.model.summary()
        self.trainable_variables = self.model.trainable_variables

    def predict(self, hidden_state, action):
        hidden_state = np.array([hidden_state])
        action = np.array([[1 if i == action else 0 for i in range(self.action_size)]])
        next_hidden_state, reward = self.model([hidden_state, action])
        next_hidden_state = np.array(next_hidden_state[0])
        reward = np.array(reward[0][0])
        return next_hidden_state, reward

class prediction:
    def __init__(self, observation_size, action_size):
        hidden_state = Input(shape=(observation_size))
        x = layers.Dense(units=1024, activation='relu')(hidden_state)
        x = layers.Dense(units=1024, activation='relu')(x)
        policy = layers.Dense(units=action_size, activation='softmax')(x)
        value = layers.Dense(units=1)(x)
        self.model = Model(inputs=hidden_state, outputs=[policy, value])
        # self.model.summary()
        self.trainable_variables = self.model.trainable_variables

    def predict(self, hidden_state):
        hidden_state = np.array([hidden_state])
        policy, value = self.model(hidden_state)
        policy = np.array(policy[0])
        value = np.array(value[0][0])
        return policy, value

class linner_model:
    def __init__(self, observation_size, action_size):
        self.representation = representation(observation_size)
        self.dynamics = dynamics(observation_size, action_size)
        self.prediction = prediction(observation_size, action_size)
        self.trainable_variables = self.representation.trainable_variables + \
                                   self.dynamics.trainable_variables + \
                                   self.prediction.trainable_variables
    def save_weights(self, path):
        self.representation.model.save_weights(path + "-representation.h5")
        self.dynamics.model.save_weights(path + '-dynamics.h5')
        self.prediction.model.save_weights(path + '-prediction.h5')

    def load_weights(self, path):
        self.representation.model.load_weights(path + "-representation.h5")
        self.dynamics.model.load_weights(path + '-dynamics.h5')
        self.prediction.model.load_weights(path + '-prediction.h5')
