import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, Conv1DTranspose, InputLayer, Reshape
from tensorflow.python.keras.layers import Flatten
from util import cnn_next_step, decnn_next_step

from config import DefaultConfig

config = DefaultConfig()


class CVAE(Model):
    def __init__(self, input_size=config.input_size, z_dim=config.z_dim):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.z_dim = z_dim
        # self.prior_z_mean = tf.Variable(tf.random_normal_initializer()(shape=[z_dim], dtype=tf.float32),
        #                                 trainable=True)
        self.prior_z_mean = tf.Variable(tf.zeros_initializer()(shape=[z_dim], dtype=tf.float32),
                                        trainable=True)
        self.prior_z_std = tf.Variable(tf.zeros_initializer()(shape=[z_dim], dtype=tf.float32),
                                       trainable=True)
        self.q_net_hidden_layers = tf.keras.Sequential(
            [
                InputLayer(input_shape=(self.input_size,)),
                Dense(self.input_size),
                Reshape(target_shape=(self.input_size, 1)),
                Conv1D(1, config.kernel_size_1, strides=config.stride_1, padding='valid',
                       activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001)),
                Conv1D(1, config.kernel_size_2, strides=config.stride_2, padding='valid',
                       activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001)),  # (?, 65, 1)
                Flatten(),
            ]
        )
        _in = cnn_next_step(config.kernel_size_1, config.stride_1, self.input_size)
        _in = cnn_next_step(config.kernel_size_2, config.stride_2, _in)
        self.q_z_mu = tf.keras.Sequential(
            [
                InputLayer(input_shape=(_in + config.cluster_num,)),
                Dense(_in + config.cluster_num),
                Dense(self.z_dim),
            ]
        )
        self.q_z_std = tf.keras.Sequential(
            [
                InputLayer(input_shape=(_in + config.cluster_num,)),
                Dense(_in + config.cluster_num),
                Dense(self.z_dim, activation='softplus'),
            ]
        )
        self.p_net_hidden_layers = tf.keras.Sequential(
            [
                InputLayer(input_shape=(self.z_dim,)),
                Dense(self.z_dim),
                Reshape(target_shape=(self.z_dim, 1)),
                Conv1DTranspose(1, config.kernel_size_2, strides=config.stride_2, padding='valid',
                                activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001)),
                Conv1DTranspose(1, config.kernel_size_1, strides=config.stride_1, padding='valid',
                                activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001)),
                Flatten(),
            ]
        )
        _out = decnn_next_step(config.kernel_size_2, config.stride_2, self.z_dim)
        _out = decnn_next_step(config.kernel_size_1, config.stride_1, _out)
        self.p_x_mu = tf.keras.Sequential(
            [
                InputLayer(input_shape=(_out + config.cluster_num,)),
                Dense(_out + config.cluster_num),
                Dense(input_size),
            ]
        )
        self.p_x_std = tf.keras.Sequential(
            [
                InputLayer(input_shape=(_out + config.cluster_num,)),
                Dense(_out + config.cluster_num),
                Dense(input_size, activation='softplus'),
            ]
        )

    @tf.function
    def reconstruct_x(self, data):
        z_mean, z_std, c, x = self.encode(data)
        x_mean, x_std = self.decode(z_mean, c)
        rec_x = tfp.distributions.Normal(x_mean, x_std)
        logpx_z = tf.reduce_sum(rec_x.log_prob(x), axis=1)
        return x_mean, x_std, c, logpx_z

    def encode(self, data):
        x, c = tf.split(data, [config.input_size, config.cluster_num], axis=-1)
        h_x = self.q_net_hidden_layers(x)
        h_z = tf.concat([h_x, c], axis=-1)
        z_mean, z_std = self.q_z_mu(h_z), self.q_z_std(h_z)
        return z_mean, z_std, c, x

    def decode(self, z, c):
        h_z = self.p_net_hidden_layers(z)
        h_x = tf.concat([h_z, c], axis=-1)
        x_mean, x_std = self.p_x_mu(h_x), self.p_x_std(h_x)
        return x_mean, x_std

    def priorz(self):
        priorz_mu, priorz_std = self.prior_z_mean, self.prior_z_std
        # add a constant lower-bound(1.) to the std of prior p(z|c) and let it be an unit gaussion at least
        priorz_std = 1. + tf.nn.softplus(priorz_std)
        # priorz_std = tf.nn.softplus(priorz_std)
        return priorz_mu, priorz_std
