from datetime import datetime


class DefaultConfig(object):
    # dir path
    data_path = '/../Dataset/data/'
    result_path = '/../result/'

    kpi_num = 18
    day_num = 45
    history_day_num = 7
    input_size = 288  # time_length
    cluster_num = 3
    z_dim = 5
    kernel_size_1 = 6  # 12
    kernel_size_2 = 3  # 6
    stride_1 = 2  # 4
    stride_2 = 1  # 3

    max_epoch = 15
    lr = 1e-3
    batch_size = 18

    # lr_decay = 0.95  # when val_loss increase, lr = lr * lr_decay
    # weight_decay = 1e-10

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/gradient_tape/' + current_time
