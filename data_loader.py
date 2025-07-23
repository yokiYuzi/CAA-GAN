# data_loader.py

import torch
import numpy as np
from torch.utils.data import Dataset
from Utils.DataUtils import DataUtils
from Utils.TrainUtils import TrainUtils
from sklearn.preprocessing import MinMaxScaler
# 导入我们需要的 train_test_split 函数
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



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
class Data_Loader():
    """
    数据加载和处理的总指挥。
    这个类现在负责按比例分割数据集。
    """
    def __init__(self):
        self.dataUtils = DataUtils()
        self.trainUtils = TrainUtils()
        
    def get_split_data(self, test_size=0.2, random_state=42):
        """
        一个全新的方法，用于加载、处理并按比例分割数据。
        """
        print("正在加载和预处理所有数据...")
        ecg_windows, fecg_windows, fqrs_rpeaks = self.trainUtils.prepareData(delay=5)
        
        print(f"数据加载完成，总样本数: {len(ecg_windows)}")
        print(f"即将按照 8:2 的比例分割数据集...")

        # 使用 train_test_split 函数进行分割
        X_train, X_test, Y_train, Y_test, fqrs_train, fqrs_test = train_test_split(
            ecg_windows,
            fecg_windows,
            fqrs_rpeaks,
            test_size=test_size,
            random_state=random_state
        )

        print("-" * 30)
        print("数据集分割完成:")
        print(f"训练集样本数: {len(X_train)}")
        print(f"测试集样本数: {len(X_test)}")
        print("-" * 30)

        # 返回分割好的六个部分
        return X_train, X_test, Y_train, Y_test, fqrs_train, fqrs_test


class FECGDataset(Dataset):
    """
    自定义数据集类，用于加载和预处理FECG数据。
    """
    def __init__(self, data_item, train=True):
        """
        初始化数据集。
        """
        super(FECGDataset, self).__init__()
        self.train = train
        
        if self.train:
            # 在训练模式下，使用训练数据集
            self.signals = data_item.X_train
            self.labels = data_item.Y_train
            self.fqrs_rpeaks = data_item.fqrs_train # 现在 data_item 有 fqrs_train 了
            print(f"数据集已初始化为 [训练模式]，包含 {len(self.signals)} 个样本。")
        else:
            # 在测试模式下，使用测试数据集
            self.signals = data_item.X_test
            self.labels = data_item.Y_test
            self.fqrs_rpeaks = data_item.fqrs_test # 同样，为测试集也准备好 fqrs
            print(f"数据集已初始化为 [测试模式]，包含 {len(self.signals)} 个样本。")

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.signals)

    def __getitem__(self, index):
        """
        根据索引获取一个数据样本。
        """
        if self.train:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    # get_train_item, get_test_item, find_normalization_range 等方法保持不变...
    def get_train_item(self, index):
        """获取并处理一个训练样本。"""
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
        """获取并处理一个测试样本。"""
        original_xx = self.signals[index, :, :]
        original_yy = self.labels[index, :, :]
        scaler_standard = MinMaxScaler(feature_range=(-1, 1), copy=False)
        AECG_signal_normalized = scaler_standard.fit_transform(original_xx.transpose()).transpose()
        return (torch.from_numpy(AECG_signal_normalized).type(torch.FloatTensor),
                torch.from_numpy(original_yy).type(torch.FloatTensor),
                torch.tensor(0))

    def find_normalization_range(self, xx, fqrs):
        """一个辅助函数，用于执行寻找归一化范围的逻辑。"""
        t_max = 10000
        t_min = 0
        if isinstance(fqrs, (list, np.ndarray)) and len(fqrs) > 0: # 增加一个检查，防止fqrs为空
            for coo in fqrs:
                start, end = max(coo - 10, 0), min(coo + 10, 127)
                if start < end: # 确保切片有效
                    v_max = np.max(xx[0, start:end])
                    if t_max > v_max:
                        t_max = v_max
                        t_min = np.min(xx[0, start:end])
        # 如果fqrs为空或无效，返回一个默认范围
        if t_max == 10000:
            return np.min(xx), np.max(xx)
        return t_min, t_max