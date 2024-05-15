import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib


def paa(series, paa_segment_size, sax_type='unidim'):
    """
    PAA implementation from

    https://github.com/seninp/saxpy/blob/master/saxpy/paa.py

    """

    series = np.array(series)
    series_len = series.shape[0]

    if sax_type in ['repeat', 'energy']:
        num_dims = series.shape[1]
    else:
        num_dims = 1
        is_multidimensional = (len(series.shape) > 1) and (series.shape[1] > 1)
        if not is_multidimensional:
            series = series.reshape(series.shape[0], 1)

    res = np.zeros((num_dims, paa_segment_size))

    for dim in range(num_dims):
        # Check if we can evenly divide the series.
        if series_len % paa_segment_size == 0:
            inc = series_len // paa_segment_size

            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res[dim], idx, np.mean(series[i][dim]))
            res[dim] /= inc
        # Process otherwise.
        else:
            for i in range(0, paa_segment_size * series_len):
                idx = i // series_len
                pos = i // paa_segment_size
                np.add.at(res[dim], idx, np.mean(series[pos][dim]))
            res[dim] /= series_len

    if sax_type in ['repeat', 'energy']:
        return res.T
    else:
        return res.flatten()


def RelativePositionMatrix(x, k):
    """
    input：
    x: 一维 时间序列
    k： 分段聚合近似(PAA)的缩减因子

    """

    x = np.squeeze(x)

    mu = np.mean(x)
    std_dev = np.std(x)
    z = (x - mu) / std_dev

    m = int(np.ceil(len(x) / k))
    # PAA
    X = paa(z, m)

    temp = matlib.repmat(X, m, 1)
    M = temp - temp.T
    RPM = 255 * (M - np.min(M)) / (np.max(M) - np.min(M))
    return RPM


# 读取CSV文件
data = pd.read_csv('../data/UCRArchive_2018/Car/Car_TRAIN.tsv', nrows=1, sep='\t', header=None)  # 只读取第一行

# 提取第一行数据（除了列名）
row_data = data.iloc[0].values

# 第一个数为label，其他为观测值
label = row_data[0]
observations = row_data[1:]

# 使用折线图进行可视化（从第二个元素开始）
plt.figure(figsize=(10, 5))  # 设置图形大小
plt.plot(range(1, len(observations) + 1), observations, marker='', color='red', linewidth='4')  # 从第二个元素开始绘制折线图
# plt.title(f'Observations for Label: {label}')  # 设置图表标题
plt.xlabel([])  # 设置x轴标签
plt.ylabel([])  # 设置y轴标签
# plt.grid(True)  # 显示网格线
# 隐藏坐标轴
plt.axis('off')
plt.savefig('../pic/ts_vision.svg', bbox_inches='tight')
plt.show()  # 显示图表

img = RelativePositionMatrix(observations, 8)
plt.figure()
# plt.close('all')
# plt.axis('off')
plt.margins(0, 0)
plt.imshow(img, cmap='viridis', origin='lower')  # 可以更换camp的颜色
plt.axis('off')
plt.savefig('RPM4.svg', bbox_inches='tight')  # 可以保存其他格式，jpg，png
