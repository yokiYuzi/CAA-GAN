from sklearn.model_selection import train_test_split
from Utils.DataUtils import DataUtils
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
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
    该类根据是训练模式还是测试模式，执行不同的数据处理流程。
    """
    def __init__(self, data_item, train=True):
        """
        初始化数据集。

        参数:
        - data_item: 包含训练和测试数据的对象。
        - train (bool): 标志位，True表示训练模式，False表示测试模式。
        """
        super(FECGDataset, self).__init__()
        self.train = train
        
        # 将数据源清晰地分离
        if self.train:
            # 在训练模式下，使用训练数据集
            self.signals = data_item.X_train
            self.labels = data_item.Y_train
            self.fqrs_rpeaks = data_item.fqrs_rpeaks
            print(f"数据集已初始化为 [训练模式]，包含 {len(self.signals)} 个样本。")
        else:
            # 在测试模式下，使用测试数据集
            self.signals = data_item.X_test
            self.labels = data_item.Y_test
            print(f"数据集已初始化为 [测试模式]，包含 {len(self.signals)} 个样本。")

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.signals)

    def __getitem__(self, index):
        """
        根据索引获取一个数据样本。
        根据是训练还是测试模式，执行不同的数据增强和归一化逻辑。
        """
        if self.train:
            # --- 训练模式下的数据处理 ---
            return self.get_train_item(index)
        else:
            # --- 测试模式下的数据处理 ---
            return self.get_test_item(index)

    def get_train_item(self, index):
        """
        获取并处理一个训练样本。
        此方法包含复杂的信号处理逻辑，用于生成训练所需的各种信号。
        """
        # 1. 获取原始信号
        xx = self.signals[index, :, :]
        yy = self.labels[index, :, :]
        fqrs = self.fqrs_rpeaks[index]

        # 2. 一套复杂的逻辑，用于寻找合适的归一化范围 (t_min, t_max)
        # 这部分是原始代码的核心，我们保留其功能并简化结构
        t_min, t_max = self.find_normalization_range(xx, fqrs)

        # 检查计算出的范围是否有效，如果无效，则换一个样本
        if t_max == np.max(xx[0, :]) or t_min == np.min(xx[0, :]):
            # 如果范围无效，尝试获取相邻的下一个样本（如果到了末尾则取上一个）
            new_index = (index + 1) if (index + 1 < len(self.signals)) else (index - 1)
            xx = self.signals[new_index, :, :]
            yy = self.labels[new_index, :, :]
            fqrs = self.fqrs_rpeaks[new_index]
            t_min, t_max = self.find_normalization_range(xx, fqrs)

        # 3. 根据计算出的动态范围归一化 yy
        scaler_dynamic = MinMaxScaler(feature_range=(t_min, t_max), copy=False)
        yy_minmax = scaler_dynamic.fit_transform(yy.transpose())

        # 4. 生成母体心电信号 (MECG)
        MECG_signal = xx - yy_minmax.transpose()

        # 5. 生成噪声信号 (BIAS)
        noise = MECG_signal.copy()
        M_index = np.argmax(MECG_signal, axis=-1)
        # 将MECG峰值附近区域平滑化以作为噪声
        start, end = int(max(0, M_index - 15)), int(min(MECG_signal.shape[-1], M_index + 15))
        noise[0, start:end] = noise[0, start]

        # 6. 将所有信号归一化到标准的 (-1, 1) 范围
        scaler_standard = MinMaxScaler(feature_range=(-1, 1), copy=False)
        
        # 使用原始的 xx 和 yy 进行归一化，而不是使用处理过的
        original_xx = self.signals[index, :, :]
        original_yy = self.labels[index, :, :]
        
        AECG_signal = scaler_standard.fit_transform(original_xx.transpose()).transpose()
        FECG_signal = scaler_standard.fit_transform(original_yy.transpose()).transpose()
        MECG_signal_norm = scaler_standard.fit_transform(MECG_signal.transpose()).transpose()
        BIAS_signal_norm = scaler_standard.fit_transform(noise.transpose()).transpose()
        
        # 7. 转换为Tensor并返回
        return (torch.from_numpy(AECG_signal).type(torch.FloatTensor),
                torch.from_numpy(FECG_signal).type(torch.FloatTensor),
                torch.from_numpy(MECG_signal_norm).type(torch.FloatTensor),
                torch.from_numpy(BIAS_signal_norm).type(torch.FloatTensor))

    def get_test_item(self, index):
        """
        获取并处理一个测试样本。
        测试流程简单得多，只需将输入和标签归一化到 (-1, 1) 即可。
        """
        # 1. 获取原始信号
        xx = self.signals[index, :, :]
        yy = self.labels[index, :, :]

        # 2. 将信号归一化到标准的 (-1, 1) 范围
        scaler_standard = MinMaxScaler(feature_range=(-1, 1), copy=False)
        AECG_signal = scaler_standard.fit_transform(xx.transpose()).transpose()
        FECG_signal = scaler_standard.fit_transform(yy.transpose()).transpose()

        # 3. 转换为Tensor并返回
        # 为了与训练时的数据结构保持一致，我们用0作为占位符
        return (torch.from_numpy(AECG_signal).type(torch.FloatTensor),
                torch.from_numpy(FECG_signal).type(torch.FloatTensor),
                torch.tensor(0), # 占位符
                torch.tensor(0)) # 占位符

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