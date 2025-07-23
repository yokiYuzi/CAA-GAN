from sklearn.model_selection import train_test_split
from Utils.DataUtils import DataUtils
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch


import matplotlib.pyplot as plt





class TrainUtils:
    def __init__(self) -> None:
        super().__init__()
        self.dataUtils = DataUtils()

    def prepareData(self, delay=5):
        ecgAll, fecg, fqrs_rpeaks = self.dataUtils.readData(1)
        ecgAll = ecgAll[range(1), :]
        delayNum = ecgAll.shape[0]
        fecgAll = self.dataUtils.createDelayRepetition(fecg, delayNum, delay)
        for i in range(2, 5):
            ecg, fecg,fqrs_rpeaks1 = self.dataUtils.readData(i)
            ecg = ecg[range(1), :]
            
            fqrs_rpeaks1 = fqrs_rpeaks1 + 60000*(i-1)
            fqrs_rpeaks= np.append(fqrs_rpeaks,fqrs_rpeaks1)
            fecgDelayed = self.dataUtils.createDelayRepetition(fecg, delayNum, delay)
            ecgAll = np.append(ecgAll, ecg, axis=1)
            fecgAll = np.append(fecgAll, fecgDelayed, axis=1)

        ecgWindows, fecgWindows, fqrs_rpeaks = self.dataUtils.windowingSig(ecgAll, fecgAll, fqrs_rpeaks, windowSize=128)
        # fecgWindows = self.dataUtils.adaptFilterOnSig(ecgWindows, fecgWindows)
        # ecgWindows = self.dataUtils.calculateICA(ecgWindows, component=2)
        return ecgWindows, fecgWindows, fqrs_rpeaks

    def trainTestSplit(self, sig, label, trainPercent, shuffle=False):
        X_train, X_test, y_train, y_test = train_test_split(sig, label, train_size=trainPercent, shuffle=shuffle)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test




class Data_Loader():
    def __init__(self, FECG=True):
        super().__init__()
        self.trainUtils = TrainUtils()

    def test_trainSignal(self):
        ecgWindows, fecgWindows,fqrs_rpeaks = self.trainUtils.prepareData(delay=5)
        X_train, X_test, Y_train, Y_test = self.trainUtils.trainTestSplit(ecgWindows, fecgWindows, len(ecgWindows)-1)

        X_train = np.reshape(X_train, [-1, X_train.shape[2], X_train.shape[1]])
        X_test = np.reshape(X_test, [-1, X_test.shape[2], X_test.shape[1]])
        Y_train = np.reshape(Y_train, [-1, Y_train.shape[2], Y_train.shape[1]])
        Y_test = np.reshape(Y_test, [-1, Y_test.shape[2], Y_test.shape[1]])
        # print(X_train.shape)
        
        
        return X_train, X_test, Y_train, Y_test, fqrs_rpeaks

class Data_Item():
    def __init__(self, FECG=True):
        super().__init__()
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.fqrs_rpeaks = Data_Loader().test_trainSignal()

    
    
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
            self.fqrs_rpeaks = data_item.fqrs_train # <--- 使用分割后的 fqrs_train
            print(f"数据集已初始化为 [训练模式]，包含 {len(self.signals)} 个样本。")
        else:
            # 在测试模式下，使用测试数据集
            self.signals = data_item.X_test
            self.labels = data_item.Y_test
            # 测试时通常不需要fqrs，但为了结构完整性，可以保留
            # self.fqrs_rpeaks = data_item.fqrs_test 
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

    def get_train_item(self, index):
        """
        获取并处理一个训练样本。
        """
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
        """
        获取并处理一个测试样本。
        """
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
        for coo in fqrs:
            start, end = max(coo - 10, 0), min(coo + 10, 127)
            v_max = np.max(xx[0, start:end])
            if t_max > v_max:
                t_max = v_max
                t_min = np.min(xx[0, start:end])
        return t_min, t_max
