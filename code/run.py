import time
import tensorflow as tf
from util import load_cvae_data
from config import DefaultConfig
from cvae import CVAE
from loss import cvae_loss

config = DefaultConfig()
optimizer = tf.keras.optimizers.Adam(config.lr)
train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def compute_apply_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = cvae_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return loss


def main(day, start_machine, machine_num):
    model = CVAE()
    epochs = config.max_epoch
    train_dataset = load_cvae_data(day=day, start_machine=start_machine, machine_num=machine_num)
    train_summary_writer = tf.summary.create_file_writer(config.log_dir)
    for epoch in range(1, epochs + 1):
        train_loss.reset_states()
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
        end_time = time.time()
        print("Epoch {}: Loss: {}, time elapse for current epoch {}".format(epoch, train_loss.result(),
                                                                            end_time - start_time))
    # save model weights
    model.save_weights('model_weight/cvae/my_model_weight')


if __name__ == '__main__':

    day = 35
    start_machine = 0
    machine_num = 200

    main(day=day, start_machine=start_machine, machine_num=machine_num)

    model = CVAE()
    model.load_weights('model_weight/cvae/my_model_weight')
