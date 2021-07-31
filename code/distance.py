import numpy as np
import tensorflow as tf
import scipy.spatial.distance as dis
from scipy import signal

from config import DefaultConfig

config = DefaultConfig()


def cal_normal_pdf(res, std=0.11):
    # return 1 / (std * (2. * np.pi) ** 0.5) * tf.math.exp(- (sample - mean) ** 2 / (2 * std ** 2))
    temp_score = tf.math.exp(- res ** 2 / (2 * std ** 2)).numpy()

    return temp_score


def distance_score(ori, rec, window_size=11):
    res = np.abs(ori - rec)
    res_cut_soft = np.maximum(res - 0.05, 0)
    # Median Filter
    res_med = signal.medfilt(res_cut_soft, window_size)

    # cal Manhattan distance directly
    zero_line = np.zeros_like(res_med)
    score = dis.minkowski(res_med, zero_line, 1)

    # cal Manhattan distance through gaussian map
    # gaussian_score = cal_normal_pdf(res_med)
    # one_line = np.ones_like(res_med)
    # score = dis.minkowski(gaussian_score, one_line, 1)

    return score
