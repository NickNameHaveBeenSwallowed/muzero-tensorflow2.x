from tensorflow.keras import layers, Input, Model, regularizers
import tensorflow as tf
import numpy as np

num_blocks = 6
l2 = 1e-5

class ResidualBlock(Model):
    def __init__(self, num_channels, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters=num_channels, kernel_size=3, strides=strides,
                                   padding='SAME', use_bias=False,
                                   kernel_regularizer=regularizers.l2(l2))
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters=num_channels, kernel_size=3, strides=strides,
                                   padding='SAME', use_bias=False,
                                   kernel_regularizer=regularizers.l2(l2))
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.add([inputs, x])
        return tf.nn.relu(x)

class LinearRseidual(Model):
    def __init__(self, units):
        super(LinearRseidual, self).__init__()
        self.linear1 = layers.Dense(units=units, use_bias=False, kernel_regularizer=regularizers.l2(l2))
        self.bn1 = layers.BatchNormalization()
        self.linear2 = layers.Dense(units=units, use_bias=False, kernel_regularizer=regularizers.l2(l2))
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.linear1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = layers.add([inputs, x])
        return tf.nn.relu(x)

class representation:
    def __init__(self, observation_shape, hidden_state_channel):
        observation = Input(shape=observation_shape)
        x = layers.Conv2D(filters=hidden_state_channel, kernel_size=3, strides=1,
                          padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(l2))(observation)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        for _ in range(num_blocks):
            x = ResidualBlock(hidden_state_channel)(x)

        hidden_state = x

        self.model = Model(inputs=observation, outputs=hidden_state)
        self.trainable_variables = self.model.trainable_variables

    def predict(self, observation):
        observation = np.array([observation])
        hidden_state = np.array(self.model(observation)[0])
        return hidden_state

class dynamics:
    def __init__(self, hidden_state_shape, hidden_state_channel, num_chess):
        self.num_chess = num_chess
        hidden_state = Input(shape=hidden_state_shape)
        action = Input(shape=(num_chess, num_chess, 1))
        x = layers.Flatten()(action)
        x = layers.Dense(units=num_chess * num_chess * hidden_state_channel, use_bias=False,
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        for _ in range(num_blocks // 2):
            x = LinearRseidual(units=num_chess * num_chess * hidden_state_channel)(x)

        x = tf.reshape(x, shape=(-1, num_chess, num_chess, hidden_state_channel))
        x = layers.add([hidden_state, x])

        for _ in range(num_blocks // 2):
            x = ResidualBlock(hidden_state_channel)(x)
        next_hidden_state = x

        self.model = Model(inputs=[hidden_state, action], outputs=next_hidden_state)
        self.trainable_variables = self.model.trainable_variables

    def predict(self, hidden_state, action):
        hidden_state = np.array([hidden_state])
        action = np.array([1 if i == action else 0 for i in range(self.num_chess ** 2)])
        action = np.reshape(action, newshape=(1, self.num_chess, self.num_chess, 1))
        next_hidden_state = self.model([hidden_state, action])
        next_hidden_state = np.array(next_hidden_state[0])
        return next_hidden_state

class prediction:
    def __init__(self, hidden_state_shape, hidden_state_channel, num_chess):
        hidden_state = Input(shape=hidden_state_shape)
        x = hidden_state
        for _ in range(num_blocks):
            x = ResidualBlock(hidden_state_channel)(x)

        policy = layers.Conv2D(filters=32, kernel_size=3, strides=1,
                          padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(l2))(x)
        policy = layers.BatchNormalization()(policy)
        policy = layers.Activation('relu')(policy)
        policy = layers.Flatten()(policy)
        policy = layers.Dense(units=1024, activation='relu', kernel_regularizer=regularizers.l2(l2))(policy)
        policy = layers.Dense(units=num_chess ** 2, activation='softmax', kernel_regularizer=regularizers.l2(l2))(policy)

        value = layers.Conv2D(filters=32, kernel_size=3, strides=1,
                          padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(l2))(x)
        value = layers.BatchNormalization()(value)
        value = layers.Activation('relu')(value)
        value = layers.Flatten()(value)
        value = layers.Dense(units=1024, activation='relu', kernel_regularizer=regularizers.l2(l2))(value)
        value = layers.Dense(units=1, activation='tanh', kernel_regularizer=regularizers.l2(l2))(value)
        self.model = Model(inputs=hidden_state, outputs=[policy, value])
        self.trainable_variables = self.model.trainable_variables

    def predict(self, hidden_state):
        hidden_state = np.array([hidden_state])
        policy, value = self.model(hidden_state)
        policy = np.array(policy[0])
        value = np.array(value[0][0])
        return policy, value

class model:
    def __init__(self, observation_shape, hidden_state_channel, num_chess):
        self.representation = representation(observation_shape, hidden_state_channel)
        hidden_state_shape = (observation_shape[0], observation_shape[1], hidden_state_channel)
        self.dynamics = dynamics(hidden_state_shape, hidden_state_channel, num_chess)
        self.prediction = prediction(hidden_state_shape, hidden_state_channel, num_chess)
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

    def copy_weights(self, target_model):
        self.representation.model.set_weights(target_model.representation.model.get_weights())
        self.dynamics.model.set_weights(target_model.dynamics.model.get_weights())
        self.prediction.model.set_weights(target_model.prediction.model.get_weights())