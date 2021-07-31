import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
from sklearn.cluster import AgglomerativeClustering
from config import DefaultConfig

config = DefaultConfig()


def cnn_next_step(kernel_size, stride, input_size):
    return int((input_size - kernel_size) / stride + 1)


def decnn_next_step(kernel_size, stride, input_size):
    return int((input_size - 1) * stride + kernel_size)


def normal_pdf(sample, mean, std):
    # return 1 / (std * (2. * np.pi) ** 0.5) * tf.math.exp(- (sample - mean) ** 2 / (2 * std ** 2))
    return tf.math.exp(- (sample - mean) ** 2 / (2 * std ** 2))


def load_plane_data(day=35, start_machine=0, machine_num=100, is_train=False):
    plane_data = np.zeros(shape=(config.kpi_num * machine_num, config.input_size)).astype('float32')
    for machine_id in range(start_machine, start_machine + machine_num):
        path = os.getcwd() + config.data_path + '{machine_id}.txt'.format(machine_id=machine_id)
        plane_data_per_machine = np.loadtxt(path, delimiter=',', skiprows=1 + (day - 1) * config.input_size,
                                            max_rows=config.input_size).astype(np.float32)
        plane_data[(machine_id - start_machine) * config.kpi_num:
                   (machine_id - start_machine + 1) * config.kpi_num, :] = plane_data_per_machine.T

    hac = AgglomerativeClustering(n_clusters=config.cluster_num).fit(plane_data)
    cluster_result = hac.labels_
    cluster_result_one_hot = np.eye(config.cluster_num)[cluster_result].astype('float32')
    data = np.hstack((plane_data, cluster_result_one_hot))

    # save cluster result
    # result = cluster_result.reshape(machine_num, config.kpi_num).T
    # data_df = pd.DataFrame(result)
    # data_df.columns = [i for i in range(start_machine, start_machine + machine_num)]
    # data_df.index = [i for i in range(config.kpi_num)]
    # writer = pd.ExcelWriter('hac.xlsx')
    # data_df.to_excel(writer, 'page_1', float_format='%d')
    # writer.save()

    if is_train:
        return cluster_result_one_hot
    else:
        return tf.convert_to_tensor(data)


def load_cvae_data(day=35, start_machine=0, machine_num=100):
    cluster_result_one_hot = load_plane_data(day=day, start_machine=start_machine, machine_num=machine_num,
                                             is_train=True)
    data = np.zeros(shape=(config.kpi_num * machine_num * config.history_day_num,
                           (config.input_size + config.cluster_num))).astype('float32')
    for machine_id in range(start_machine, start_machine + machine_num):
        path = os.getcwd() + config.data_path + '{machine_id}.txt'.format(machine_id=machine_id)
        temp_data = np.loadtxt(path, delimiter=',', skiprows=1 + (day - config.history_day_num - 1) * config.input_size,
                               max_rows=config.input_size * config.history_day_num).astype(np.float32)
        for kpi_id in range(config.kpi_num):
            c_one_hot = cluster_result_one_hot[(machine_id - start_machine) * config.kpi_num + kpi_id]
            for day_id in range(config.history_day_num):
                start = day_id * config.input_size
                end = start + config.input_size
                x_con_c = np.hstack((temp_data[start:end, kpi_id], c_one_hot))
                data[(kpi_id * config.history_day_num + day_id) +
                     (machine_id - start_machine) * config.kpi_num * config.history_day_num] = x_con_c
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    dataset = tf_data.shuffle(config.kpi_num * machine_num * config.history_day_num).batch(config.batch_size)

    return dataset
