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

    def load_and_split_data(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        # 确保比例合规
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError("train/val/test ratios must sum to 1.0")

        ecgWindows, fecgWindows, fqrs_rpeaks = self.trainUtils.prepareData(delay=5)
        
        # 将数据转换为numpy数组以便索引
        ecgWindows = np.array(ecgWindows)
        fecgWindows = np.array(fecgWindows)
        fqrs_rpeaks = np.array(fqrs_rpeaks, dtype=object) # fqrs是不等长列表，dtype=object

        # 第一次分割 -> 训练集 vs (验证集+测试集)
        X_train, X_temp, Y_train, Y_temp, fqrs_train, fqrs_temp = train_test_split(
            ecgWindows, fecgWindows, fqrs_rpeaks, train_size=train_ratio, shuffle=False
        )

        # 计算第二次分割的比例
        val_test_ratio = val_ratio / (val_ratio + test_ratio)

        # 第二次分割 -> 验证集 vs 测试集
        X_val, X_test, Y_val, Y_test, fqrs_val, fqrs_test = train_test_split(
            X_temp, Y_temp, fqrs_temp, train_size=val_test_ratio, shuffle=False
        )
        
        # 调整形状 (N, C, L)
        def reshape_data(arr):
            return np.reshape(arr, [-1, arr.shape[2], arr.shape[1]])

        X_train, Y_train = reshape_data(X_train), reshape_data(Y_train)
        X_val, Y_val = reshape_data(X_val), reshape_data(Y_val)
        X_test, Y_test = reshape_data(X_test), reshape_data(Y_test)

        print(f"Data split completed:")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return (X_train, Y_train, fqrs_train), (X_val, Y_val, fqrs_val), (X_test, Y_test, fqrs_test)


class Data_Item():
    def __init__(self, FECG=True):
        super().__init__()
        # 获取分割后的数据
        (self.X_train, self.Y_train, self.fqrs_train), \
        (self.X_val, self.Y_val, self.fqrs_val), \
        (self.X_test, self.Y_test, self.fqrs_test) = Data_Loader().load_and_split_data()

    
    
class FECGDataset(Dataset):
    """
    一个用于处理胎儿心电图（FECG）数据的 PyTorch 数据集类。
    
    该类的主要任务是接收腹部心电图（AECG）和作为标签的胎儿心电图（FECG），
    并动态地生成母体心电图（MECG）和基线漂移/噪声（BIAS）作为额外的训练目标。
    """
    def __init__(self, data, labels, fqrs):
        """
        构造函数。
        
        参数:
            data (np.array): 输入数据，通常是腹部心电图(AECG)信号。形状为 (N, C, L)。
            labels (np.array): 标签数据，通常是纯净的胎儿心电图(FECG)信号。形状为 (N, C, L)。
            fqrs (np.array): 胎儿R波峰值的位置索引列表。形状为 (N, )。
        """
        super(FECGDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.fqrs = fqrs
        self.num_samples = len(self.data)

    def _process_signals(self, index):
        """
        核心处理函数，根据给定的索引处理单个样本。
        它实现了动态生成MECG和BIAS信号的复杂逻辑。
        """
        # 步骤 1: 获取原始信号
        aecg_signal = self.data[index, :, :]    # (C, L)
        fecg_signal = self.labels[index, :, :]  # (C, L)
        fqrs_locations = self.fqrs[index]       # list of ints

        # 步骤 2: 基于胎儿R波位置，在AECG中估计FECG的振幅范围 (t_min, t_max)
        # 这是一个关键的假设：AECG中对应胎儿R波位置的振幅可以用来缩放标准FECG信号
        t_max = 10000.0  # 初始化一个较大的值
        t_min = 0.0
        
        # 检查fqrs_locations是否为空或无效
        if fqrs_locations and len(fqrs_locations) > 0:
            for coo in fqrs_locations:
                # 定义一个围绕R波峰值的小窗口
                start = max(coo - 10, 0)
                end = min(coo + 10, aecg_signal.shape[1] - 1)
                if start >= end: continue # 如果窗口无效则跳过
                
                v_max = np.max(aecg_signal[0, start:end])
                if t_max > v_max:
                    t_max = v_max
                    t_min = np.min(aecg_signal[0, start:end])
        
        # 步骤 3: Fallback机制
        # 如果上一步未能找到合适的振幅范围（例如，t_max仍然是全局最大值），
        # 则尝试使用相邻的样本进行处理。这可以增加数据处理的鲁棒性。
        if t_max == np.max(aecg_signal[0, :]) or t_min == np.min(aecg_signal[0, :]) or not fqrs_locations:
            # 选择前一个或后一个样本的索引
            fallback_index = index - 1 if (index + 1 == self.num_samples) else index + 1
            # 递归调用自身来处理这个相邻的样本
            return self._process_signals(fallback_index)

        # 步骤 4: 生成母体心电图 (MECG)
        # 使用上一步估计的振幅范围 [t_min, t_max] 来缩放真实的FECG信号
        scaler_fecg = MinMaxScaler(feature_range=(t_min, t_max), copy=True)
        # 注意：MinMaxScaler 需要 (n_samples, n_features) 格式，因此需要转置
        scaled_fecg_transposed = scaler_fecg.fit_transform(fecg_signal.transpose())
        
        # 从AECG中减去缩放后的FECG，得到估计的MECG
        mecg_signal = aecg_signal - scaled_fecg_transposed.transpose()

        # 步骤 5: 生成基线/噪声信号 (BIAS)
        # 这里的假设是，MECG信号的主要成分是母体QRS波群，其余部分可视为噪声/基线
        mecg_copy = mecg_signal.copy()
        m_index = np.argmax(mecg_copy, axis=-1)[0] # 找到母体R波峰值的大致位置
        
        # 将母体QRS波群区域“抹平”，用边界值填充，以模拟基线
        qrs_start = int(max(0, m_index - 15))
        qrs_end = int(min(mecg_copy.shape[-1], m_index + 15))
        mecg_copy[0, qrs_start:qrs_end] = mecg_copy[0, qrs_start]
        bias_signal = mecg_copy

        # 步骤 6: 归一化所有信号到 [-1, 1] 范围
        final_scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
        
        # 分别对每个信号进行归一化
        final_aecg = final_scaler.fit_transform(aecg_signal.transpose()).transpose()
        final_fecg = final_scaler.fit_transform(fecg_signal.transpose()).transpose()
        final_mecg = final_scaler.fit_transform(mecg_signal.transpose()).transpose()
        final_bias = final_scaler.fit_transform(bias_signal.transpose()).transpose()
        
        return final_aecg, final_fecg, final_mecg, final_bias

    def __getitem__(self, index):
        """
        获取并处理单个数据样本。
        """
        # 调用核心处理函数
        aecg, fecg, mecg, bias = self._process_signals(index)

        # 将所有numpy数组转换为torch.float32张量，以供模型使用
        return (
            torch.from_numpy(aecg).float(),
            torch.from_numpy(fecg).float(),
            torch.from_numpy(mecg).float(),
            torch.from_numpy(bias).float()
        )

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return self.num_samples