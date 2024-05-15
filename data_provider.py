import numpy as np
from matplotlib import pyplot as plt
from numpy import matlib
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


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


class UCR_data_provider(Dataset):
    def __init__(self, args, dataset_type: str = 'train', znorm: bool = True):
        self.input_size = 0
        self.output_size = 0
        self.data_path = args.data_path
        self.sub_data = args.sub_data
        self.dataset_type = dataset_type
        self.znorm = znorm

        self.read_dataset()

    def read_dataset(self):
        if self.dataset_type == 'train':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TRAIN.tsv', sep='\t',
                                 header=None)
        elif self.dataset_type == 'test':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TEST.tsv', sep='\t',
                                 header=None)
        else:
            raise ValueError("Illegal dataset type.")

        ts = df_raw.drop(columns=[0])
        ts = ts.fillna(0)
        self.input_size = ts.shape[1]  # shape指第index维度的维数，DataFrame的成员
        ts.columns = range(ts.shape[1])
        label = df_raw.values[:, 0]
        self.output_size = int(max(label) + 1)

        ts = ts.values

        if self.znorm:
            std_ = ts.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            ts = (ts - ts.mean(axis=1, keepdims=True)) / std_

        self.ts = ts
        self.label = label

    def __getitem__(self, index):
        ts = self.ts[index]
        label = self.label[index]
        return ts, label

    def __len__(self):
        return self.label.shape[0]


class IMG_UCR_data_provider(Dataset):
    def __init__(self, args, dataset_type: str = 'train', znorm: bool = True, transform=None):
        self.input_size = 0
        self.output_size = 0
        self.data_path = args.data_path
        self.sub_data = args.sub_data
        self.dataset_type = dataset_type
        self.znorm = znorm
        self.transform = transform

        self.read_dataset()

    def read_dataset(self):
        if self.dataset_type == 'train':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TRAIN.tsv', sep='\t',
                                 header=None)
        elif self.dataset_type == 'test':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TEST.tsv', sep='\t',
                                 header=None)
        else:
            raise ValueError("Illegal dataset type.")

        ts = df_raw.drop(columns=[0])
        self.input_size = ts.shape[1]  # shape指第index维度的维数，DataFrame的成员
        ts.columns = range(ts.shape[1])
        label = df_raw.values[:, 0]
        self.output_size = int(max(label) + 1)

        ts = ts.values

        if self.znorm:
            std_ = ts.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            ts = (ts - ts.mean(axis=1, keepdims=True)) / std_

        self.ts = ts
        self.label = label

    def __getitem__(self, index):
        ts = self.ts[index]
        img = RelativePositionMatrix(ts, 8)
        img = Image.fromarray((img * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        label = self.label[index]
        return img, label

    def __len__(self):
        return self.label.shape[0]


class M4TSC_UCR_data_provider(Dataset):
    def __init__(self, args, dataset_type: str = 'train', znorm: bool = True, transform=None):
        self.input_size = 0
        self.output_size = 0
        self.data_path = args.data_path
        self.sub_data = args.sub_data
        self.dataset_type = dataset_type
        self.znorm = znorm
        self.transform = transform

        self.image_cache = {}

        self.read_dataset()

    def read_dataset(self):
        if self.dataset_type == 'train':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TRAIN.tsv', sep='\t',
                                 header=None)
        elif self.dataset_type == 'test':
            df_raw = pd.read_csv(self.data_path + '/' + self.sub_data + '/' + self.sub_data + '_TEST.tsv', sep='\t',
                                 header=None)
        else:
            raise ValueError("Illegal dataset type.")

        ts = df_raw.drop(columns=[0])
        ts = ts.fillna(0)
        # print(ts[483][0])
        self.input_size = ts.shape[1]  # shape指第index维度的维数，DataFrame的成员
        ts.columns = range(ts.shape[1])
        label = df_raw.values[:, 0]
        self.output_size = int(max(label) + 1)

        ts = ts.values

        if self.znorm:
            std_ = ts.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            ts = (ts - ts.mean(axis=1, keepdims=True)) / std_

        self.ts = ts
        self.label = label

    def __getitem__(self, index):
        if index in self.image_cache:
            img = self.image_cache[index]
        else:
            ts = self.ts[index]
            img = RelativePositionMatrix(ts, 8)
            plt.imsave('1.png', img, cmap='bwr', origin='lower')

            # 将图像数据存入缓存
            self.image_cache[index] = img

            img = Image.fromarray((img * 255).astype(np.uint8))

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = self.label[index]
        return self.ts[index], img, label
        # ts = self.ts[index]
        # img = RelativePositionMatrix(ts, 8)
        #
        # plt.imsave('1.png', img, cmap='bwr', origin='lower')
        #
        # img = Image.fromarray((img * 255).astype(np.uint8))
        #
        # if self.transform:
        #     img = self.transform(img)
        # label = self.label[index]
        # return ts, img, label

    def __len__(self):
        return self.label.shape[0]
