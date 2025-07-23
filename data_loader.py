import torch
import numpy as np
from torch.utils.data import Dataset
from Utils.DataUtils import DataUtils
from Utils.TrainUtils import TrainUtils
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 【核心修改】只修改 Data_Item 类
class Data_Item():
    def __init__(self):
        self.trainUtils = TrainUtils()
        # 接收 prepareData 返回的5个值
        X_train_all, X_test_all, Y_train_all, Y_test_all, fqrs_tuple = self.trainUtils.prepareData(delay=5)
        
        # 将接收到的值赋给 self
        self.X_train = X_train_all
        self.X_test = X_test_all
        self.Y_train = Y_train_all
        self.Y_test = Y_test_all
        # 解开包含 fqrs 的元组
        self.fqrs_train, self.fqrs_test = fqrs_tuple


class FECGDataset(Dataset):
    # 这个类的代码现在是完全正确的，无需再修改
    def __init__(self, data_item, train=True):
        super(FECGDataset, self).__init__()
        self.train = train
        
        if self.train:
            self.signals = data_item.X_train
            self.labels = data_item.Y_train
            self.fqrs_rpeaks = data_item.fqrs_train
            print(f"数据集已初始化为 [训练模式]，包含 {len(self.signals)} 个样本。")
        else:
            self.signals = data_item.X_test
            self.labels = data_item.Y_test
            self.fqrs_rpeaks = data_item.fqrs_test
            print(f"数据集已初始化为 [测试模式]，包含 {len(self.signals)} 个样本。")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        if self.train:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)
    
    # get_train_item, get_test_item, 和 find_normalization_range 方法保持不变
    def get_train_item(self, index):
        xx = self.signals[index, :, :]
        yy = self.labels[index, :, :]
        fqrs = self.fqrs_rpeaks[index]
        t_min, t_max = self.find_normalization_range(xx, fqrs)
        if t_max == np.max(xx[0, :]) or t_min == np.min(xx[0, :]):
            new_index = (index + 1) if (index + 1 < len(self.signals)) else (index - 1)
            xx = self.signals[new_index, :, :]
            yy = self.labels[new_index, :, :]
            fqrs = self.fqrs_rpeaks[new_index]
            t_min, t_max = self.find_normalization_range(xx, fqrs)
        scaler_dynamic = MinMaxScaler(feature_range=(t_min, t_max), copy=False)
        yy_minmax = scaler_dynamic.fit_transform(yy.transpose())
        MECG_signal = xx - yy_minmax.transpose()
        noise = MECG_signal.copy()
        M_index = np.argmax(MECG_signal, axis=-1)
        start, end = int(max(0, M_index - 15)), int(min(MECG_signal.shape[-1], M_index + 15))
        noise[0, start:end] = noise[0, start]
        scaler_standard = MinMaxScaler(feature_range=(-1, 1), copy=False)
        original_xx = self.signals[index, :, :]
        original_yy = self.labels[index, :, :]
        AECG_signal = scaler_standard.fit_transform(original_xx.transpose()).transpose()
        FECG_signal = scaler_standard.fit_transform(original_yy.transpose()).transpose()
        MECG_signal_norm = scaler_standard.fit_transform(MECG_signal.transpose()).transpose()
        BIAS_signal_norm = scaler_standard.fit_transform(noise.transpose()).transpose()
        return (torch.from_numpy(AECG_signal).type(torch.FloatTensor),
                torch.from_numpy(FECG_signal).type(torch.FloatTensor),
                torch.from_numpy(MECG_signal_norm).type(torch.FloatTensor),
                torch.from_numpy(BIAS_signal_norm).type(torch.FloatTensor))

    def get_test_item(self, index):
        original_xx = self.signals[index, :, :]
        original_yy = self.labels[index, :, :]
        scaler_standard = MinMaxScaler(feature_range=(-1, 1), copy=False)
        AECG_signal_normalized = scaler_standard.fit_transform(original_xx.transpose()).transpose()
        return (torch.from_numpy(AECG_signal_normalized).type(torch.FloatTensor),
                torch.from_numpy(original_yy).type(torch.FloatTensor),
                torch.tensor(0))

    def find_normalization_range(self, xx, fqrs):
        t_max = 10000
        t_min = 0
        if isinstance(fqrs, (list, np.ndarray)) and len(fqrs) > 0:
            for coo in fqrs:
                start, end = max(coo - 10, 0), min(coo + 10, 127)
                if start < end:
                    v_max = np.max(xx[0, start:end])
                    if t_max > v_max:
                        t_max = v_max
                        t_min = np.min(xx[0, start:end])
        if t_max == 10000:
            return np.min(xx), np.max(xx)
        return t_min, t_max