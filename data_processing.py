import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def generate_graph_seq2seq_io_data(df, ratio, x_offsets, y_offsets):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)

    k = 0
    data_x = []
    data_y = []
    for i in ratio:
        num_certain = int(num_samples * i)
        x = []
        y = []
        min_t = int(k + abs(min(x_offsets)))
        max_t = int(abs(k + num_certain - abs(max(y_offsets))))
        for t in range(min_t, max_t):
            one = data[t + x_offsets, ...]
            x.append(one)
            one = data[t + y_offsets, ...][:, [3, 8, 13], :]
            y.append(one)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        data_x.append(x)
        data_y.append(y)
        k += i
    return data_x, data_y

def generate_dataset(input_file, out_dir, seq_length_x = 24, seq_length_y = 1, ratio = [0.6, 0.2, 0.2]):
    df = pd.read_csv(input_file, usecols=[0]+list(np.arange(1, 16)), encoding='gbk')
    df.columns =  ['time', 'pH_U', 'turbidity_U', 'salinity_U', 'DO_U', 'temperature_U',
                   'pH_M', 'turbidity_M', 'salinity_M', 'DO_M', 'temperature_M',
                   'pH_D', 'turbidity_D', 'salinity_D', 'DO_D', 'temperature_D']
    df.set_index(keys='time', inplace=True)
    # print(list(enumerate(df.columns)))
    exi = os.path.exists(out_dir)
    if not exi:
        os.mkdir(out_dir)
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    num_samples = df.shape[0]

    train_ratio, val_ratio, test_ratio = ratio

    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = int(num_samples * test_ratio)
    print("train:{}, val:{}, test:{}".format(num_train, num_val, num_test))

    ratio = [train_ratio, val_ratio, test_ratio]

    data_x, data_y = generate_graph_seq2seq_io_data(
        df,
        ratio,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )

    # print("x shape: ", x.shape, ", y shape: ", y.shape)
    # per = np.random.permutation(x.shape[0])
    # x = x[per]
    # y = y[per]

    x_train, y_train = data_x[0], data_y[0]
    x_val, y_val = data_x[1], data_y[1]
    x_test, y_test = data_x[2], data_y[2]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(out_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def get_adj_file(input_file, train_ratio=0.3):
    df = pd.read_csv(input_file, usecols=list(np.arange(1, 16)), encoding='gbk')
    # df.columns = ['pH_U', 'turbidity_U', 'salinity_U', 'DO_U', 'temperature_U']
    df.columns =  [ 'pH_U', 'turbidity_U', 'salinity_U', 'DO_U', 'temperature_U',
                   'pH_M', 'turbidity_M', 'salinity_M', 'DO_M', 'temperature_M',
                   'pH_D', 'turbidity_D', 'salinity_D', 'DO_D', 'temperature_D']

    num_samples, num_nodes = df.shape
    train_samples = int(num_samples * train_ratio)
    df_value = df.values[:train_samples,:]
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i,num_nodes):
            adj[i, j] = adj[j,i] = np.corrcoef(df_value[:,i], df_value[:,j])[0,1]
    # 绘制相关矩阵
    adj = pd.DataFrame(adj)
    adj.columns=df.columns
    adj.index=df.columns
    plt.figure(figsize=(12, 12))
    sns.heatmap(adj, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
    np.save('./data/multi_level_var_adj_{}'.format(train_ratio),adj) #0.3/0.5/0.7

import utils.util as util
if __name__ == "__main__":
    util.set_seed(20250523)
    seq_length_x = [48]
    seq_length_y = 12
    ratio = [0.3, 0.1, 0.2]
    for i in seq_length_x:
        generate_dataset('data/data.csv', './data/DO_multi_level_{}-{}_step_train{}/'.format(i,seq_length_y, ratio[0]),
                     i, seq_length_y, ratio)

    # generate a sample adj file
    # get_adj_file('data/data.csv', 0.7)
    # a=np.load(os.path.join('./data/DO_down_level_6-1_step/test.npz'))['x']
    # print(1)

