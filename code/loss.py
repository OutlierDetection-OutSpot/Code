import tensorflow as tf
import tensorflow_probability as tfp
from config import DefaultConfig

config = DefaultConfig()


@tf.function
def cvae_loss(model, data):
    z_mean, z_std, c, x = model.encode(data)
    z = tfp.distributions.Normal(z_mean, z_std)
    z_sample = z.sample()

    priorz_mean, priorz_std = model.priorz()
    prior_z = tfp.distributions.Normal(priorz_mean, priorz_std)
    # prior_z = tfp.distributions.Normal(0, 1)

    x_mean, x_std = model.decode(z_sample, c)
    rec_x = tfp.distributions.Normal(x_mean, x_std)

    logpz = tf.reduce_sum(prior_z.log_prob(z_sample), axis=1)
    logqz_x = tf.reduce_sum(z.log_prob(z_sample), axis=1)

    logpx_z = tf.reduce_sum(rec_x.log_prob(x), axis=1)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
