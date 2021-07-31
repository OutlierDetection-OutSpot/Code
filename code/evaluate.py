import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from matplotlib import pyplot as plt, ticker
from sklearn.metrics import precision_recall_curve, roc_curve, auc,roc_auc_score

from util import load_plane_data
from distance import distance_score
from config import DefaultConfig
from cvae import CVAE


config = DefaultConfig()


def plot_ex(data, gen):
    f, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.plot(data)
    ax.plot(gen)
    ax.set_ylim([0, 1])
    ax.set_xlim([-2, 290])
    ax.xaxis.set_major_locator(ticker.FixedLocator([i * 24 for i in range(13)]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["%02d:00" % (i * 2) for i in range(13)]))

    plt.show()


def get_ground_truth(day, start_machine, machine_num):
    file_name = "/Day" + str(day) + "_" + str(start_machine) + "-" + str(start_machine + machine_num) + ".txt"
    path = os.getcwd() + file_name
    # read the label only
    ground_truth = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(2,)).astype(np.float32)
    return ground_truth


if __name__ == '__main__':

    model = CVAE()
    model.load_weights('model_weight/cvae/my_model_weight')

    day = 35
    start_machine = 0
    machine_num = 200

    ori_data_with_cluster = load_plane_data(day=day, start_machine=start_machine, machine_num=machine_num, is_train=False)
    ori_data = tf.split(ori_data_with_cluster, [config.input_size, config.cluster_num], axis=-1)[0]
    rec_data, x_std, c, logpx_z = model.reconstruct_x(ori_data_with_cluster)
    score = np.zeros(shape=machine_num * config.kpi_num)
    for i in range(len(ori_data)):
        ori = ori_data[i].numpy()
        rec = rec_data[i].numpy()
        score[i] = distance_score(ori=ori, rec=rec)
    label = get_ground_truth(day=day, start_machine=start_machine, machine_num=machine_num)
    predict = np.ones_like(label).astype(np.int32)
    # (0.2 - 0.04) * 18(1.5h) * 80% = 2.304
    predict[score > 2.304] = 0

    # 输出错误
    print('Output the KPIs with incorrect forecast results with the threshold 2.304: \n')
    for machine_id in range(start_machine, start_machine + machine_num):
        for kpi_id in range(config.kpi_num):
            index = (machine_id - start_machine) * config.kpi_num + kpi_id
            if predict[index] != label[index]:
                print("score: {}".format(score[index]) + " == predict: {}".format(predict[index])
                      + " == truth: {}".format(label[index]))
                print("machine: {}".format(machine_id) + " == kpi: {}".format(kpi_id))
                print("-----------------------")

    print('precision: ', metrics.precision_score(y_true=label, y_pred=predict, pos_label=0))
    print('recall: ', metrics.recall_score(y_true=label, y_pred=predict, pos_label=0))
    print('f1_score: ', metrics.f1_score(y_true=label, y_pred=predict, pos_label=0))
    print('---------')

    # save score result
    result = score.reshape(machine_num, config.kpi_num).T
    data_df = pd.DataFrame(result)
    writer = pd.ExcelWriter(os.getcwd() + config.result_path + 'score.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.02f')
    writer.save()

    precision, recall, thresholds = precision_recall_curve(label, score, pos_label=0)
    f1_record = 0
    index = 0

    for i in range(len(thresholds)):
        if precision[i] == 0 or recall[i] == 0:
            continue
        f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        if f1 > f1_record:
            f1_record = f1
            index = i

    print("the threshold: ", thresholds[index])
    print("best precision: ",precision[index])
    print("best recall: ",recall[index])
    print("bets f1_score: ",f1_record)
    fpr, tpr, th = roc_curve(label, score, pos_label=0)
    auc_record = auc(fpr, tpr)
    print('AUC: ', auc_record)
    print('---------')
