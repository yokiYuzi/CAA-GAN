import pyedflib
import numpy as np
import os
# 【新增】导入官方推荐的 wfdb 库
import wfdb

DATA_ROOT_DIR = './ADFECGDB'

class DataUtils():
    def __init__(self):
        pass

    def getFileName(self, index):
        if index < 10:
            file_name = os.path.join(DATA_ROOT_DIR, 'r' + str(index).zfill(2) + '.edf')
        else:
            file_name = os.path.join(DATA_ROOT_DIR, 'r' + str(index) + '.edf')
        return file_name

    def readData(self, index):
        file_name = self.getFileName(index)
        try:
            f = pyedflib.EdfReader(file_name)
        except OSError as e:
            print(f"错误: 无法打开文件 '{file_name}'.")
            print("请确保 ADFECGDB 数据集已下载并放置在项目根目录。")
            raise e
            
        n = f.signals_in_file
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)

        ecgAll = sigbufs[0:4]
        fecg = sigbufs[4]

        # 准备用于查找标注文件的基本名
        ann_name_base = os.path.join(DATA_ROOT_DIR, 'r' + str(index).zfill(2))
        qrs_rpeaks = self.getQRS_wfdb(ann_name_base, fecg)
        return ecgAll, fecg, qrs_rpeaks

    # 【核心修正】重写 getQRS 方法，使用 wfdb 库来读取二进制标注文件
    def getQRS_wfdb(self, record_name, fecg):
        """
        使用 wfdb 库读取二进制的 .qrs 标注文件。
        record_name: 记录文件的基本路径和名称 (例如 './ADFECGDB/r01')
        """
        try:
            # wfdb.rdann 会自动寻找 .qrs 文件并正确解析
            annotation = wfdb.rdann(record_name, 'qrs')
            # Annotation 对象的 .sample 属性包含了所有R波峰值的样本索引
            qrs_rpeaks = annotation.sample
            
            # 将采样率从 1000Hz 转换为样本索引
            # 注意: 此数据集的采样率为1000Hz，所以标注文件中的时间戳乘以1000即为样本索引
            # wfdb库已经处理了这一点，直接返回的就是样本索引，无需转换
            
            return np.array(qrs_rpeaks)

        except FileNotFoundError:
            # 如果找不到 .qrs 文件，返回 None
            return None
        except Exception as e:
            print(f"使用 wfdb 读取标注文件 {record_name}.qrs 时发生错误: {e}")
            return None

    def data2window(self, data, size=128):
        shape = data.shape
        if len(shape) == 1:
            data = np.expand_dims(data, axis=0)
        shape = data.shape
        num = int(shape[1] / size)
        windows = np.zeros([num, shape[0], size])
        for i in range(num):
            windows[i, :, :] = data[:, i * size:(i + 1) * size]
        if len(shape) == 1:
            windows = windows[:, 0, :]
        return windows

    def createDelayRepetition(self, sig, delayNum, delay):
        sigDelayed = np.zeros([delayNum, sig.shape[0]])
        for i in range(delayNum):
            sigDelayed[i, :] = np.roll(sig, i * delay)
        return sigDelayed

# 一个空的占位符类，用于解决 TrainUtils.py 中的导入依赖错误
class DataUtils_NIFECGDB():
    def __init__(self):
        pass