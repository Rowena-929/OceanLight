"""
The dataloader for GenCast.

It has to:
- load, normalize and concatenate (across the channel dimension) the input timesteps 0 and 1.
- load and normalize the residual between timesteps 2 and 1.
- sample a noise level.
- corrupt the residual with noise generated at the given noise level.
"""

'''经典预报残差'''

import warnings

import einops
import numpy as np
import xarray as xr
import datetime
import h5py
from torch.utils.data import Dataset
import functools
from concurrent.futures import ThreadPoolExecutor

# from graph_weather.data import const
from utils.noise import generate_isotropic_noise, sample_noise_level
import ipdb

'''0.25度预报深层浅层一块儿'''

def load_single_file(file_path, date_str, delete_indices=None):
    """加载单个npy文件并可选地删除指定索引"""
    data = np.load(f'{file_path}mra5_{date_str}.npy')[0]
    if delete_indices is not None:
        data = np.delete(data, delete_indices, axis=0)
    return data

class TrainDataset(Dataset):
    def __init__(self, lead):
        self.file_path1 = '/fs1/home/wangxiang/data/data_1to22/train/'
        self.file_path2 = '/fs1/home/wangxiang/data/data_23to33/train/'
        self.mean = np.load('./data/normalize_mean_23_0.25deg.npz')
        self.std = np.load('./data/normalize_std_23_0.25deg.npz')
        # self.mask = np.load('/hpcfs/fhome/yangjh5/wuwei/fgn/data/land_sea_mask.npy')[:,:,::3,::3]
        self.use_isotropic_noise = False
        self.num_lon = 4320
        self.num_lat = 2041
        self.lead = lead

    def __len__(self):
        return (datetime.datetime(2017, 12, 31) - datetime.datetime(1993, 1, 1)).days - (self.lead - 1)

    def __getitem__(self, index):
        first = datetime.datetime(1993,1,1) + datetime.timedelta(days=index)
        second = first + datetime.timedelta(days=self.lead)
        # data1是前一天，data2是target值

        load_tasks = [
            (self.file_path1, first.strftime("%Y%m%d"), None),  # data1的第一部分
            (self.file_path2, first.strftime("%Y%m%d"), [0,1,2,3]),  # data1的第二部分
            (self.file_path1, second.strftime("%Y%m%d"), None),  # data2的第一部分
            (self.file_path2, second.strftime("%Y%m%d"), [0,1,2,3]),  # data2的第二部分
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda x: load_single_file(*x), load_tasks))

        data1 = np.concatenate((results[0], results[1]), axis=0)
        data1 = data1[:, ::3, ::3].astype(np.float32)
        data2 = np.concatenate((results[2], results[3]), axis=0)
        data2 = data2[:, ::3, ::3].astype(np.float32)
        data1 = np.where(data1<-30000, 0, data1)
        data2 = np.where(data2<-30000, 0, data2)
        for i in range(96):
            data1[i, :, :] = (data1[i, :, :] - self.mean[self.mean.files[i]]) / self.std[self.mean.files[i]]
            data2[i, :, :] = (data2[i, :, :] - self.mean[self.mean.files[i]]) / self.std[self.mean.files[i]]
        data1 = np.delete(data1, [1,2,3], 0)
        data2 = np.delete(data2, [1,2,3], 0)
        target_residuals = data2 - data1
        target_residuals = target_residuals.transpose(2, 1, 0)
        prev_inputs = data1.transpose(2, 1, 0)
        return (
            prev_inputs,
            target_residuals,
            index,
        )


class ValDataset(Dataset):
    def __init__(self, lead):
        self.file_path1 = '/fs1/home/wangxiang/data/data_1to22/val/'
        self.file_path2 = '/fs1/home/wangxiang/data/data_23to33/val/'
        self.mean = np.load('./data/normalize_mean_23_0.25deg.npz')
        self.std = np.load('./data/normalize_std_23_0.25deg.npz')
        # self.mask = np.load('/hpcfs/fhome/yangjh5/wuwei/fgn/data/land_sea_mask.npy')[:,:,::3,::3]
        self.use_isotropic_noise = False
        self.num_lon = 4320
        self.num_lat = 2041
        self.lead = lead

    def __len__(self):
        return 364 - (self.lead - 1)

    def __getitem__(self, index):
        first = datetime.datetime(2018, 1, 1) + datetime.timedelta(days=index)
        second = first + datetime.timedelta(days=self.lead)
        # data1是前一天，data2是target值

        load_tasks = [
            (self.file_path1, first.strftime("%Y%m%d"), None),  # data1的第一部分
            (self.file_path2, first.strftime("%Y%m%d"), [0,1,2,3]),  # data1的第二部分
            (self.file_path1, second.strftime("%Y%m%d"), None),  # data2的第一部分
            (self.file_path2, second.strftime("%Y%m%d"), [0,1,2,3]),  # data2的第二部分
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda x: load_single_file(*x), load_tasks))

        data1 = np.concatenate((results[0], results[1]), axis=0)
        data1 = data1[:, ::3, ::3].astype(np.float32)
        data2 = np.concatenate((results[2], results[3]), axis=0)
        data2 = data2[:, ::3, ::3].astype(np.float32)
        data1 = np.where(data1 < -30000, 0, data1)
        data2 = np.where(data2 < -30000, 0, data2)
        for i in range(96):
            data1[i, :, :] = (data1[i, :, :] - self.mean[self.mean.files[i]]) / self.std[
                self.mean.files[i]]
            data2[i, :, :] = (data2[i, :, :] - self.mean[self.mean.files[i]]) / self.std[
                self.mean.files[i]]
        data1 = np.delete(data1, [1, 2, 3], 0)
        data2 = np.delete(data2, [1, 2, 3], 0)
        target_residuals = data2 - data1
        target_residuals = target_residuals.transpose(2, 1, 0)
        prev_inputs = data1.transpose(2, 1, 0)
        return (
            prev_inputs.astype(np.float32),
            target_residuals.astype(np.float32),
            index,
        )