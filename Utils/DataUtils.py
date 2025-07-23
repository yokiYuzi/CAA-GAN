import pyedflib
import numpy as np
import os

DATA_ROOT_DIR = './ADFECGDB'

class DataUtils():
    def __init__(self):
        pass

    def getFileName(self, index):
        # 使用 zfill(2) 来正确格式化文件名，例如 r1 -> r01
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
        ann_name = file_name.replace(".edf", "")
        qrs_rpeaks = self.getQRS(ann_name, fecg)
        return ecgAll, fecg, qrs_rpeaks

    # 【核心修正】只修改 getQRS 这一个方法
    def getQRS(self, ann_name, fecg):
        # 之前修正的文件名拼接是正确的
        path = ann_name + '.edf.qrs'
        
        if not os.path.exists(path):
            return None
            
        qrs_rpeaks = []
        try:
            # 【关键修正】在打开文件时，明确指定编码为 'latin-1'
            with open(path, "r", encoding="latin-1") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        str_time = parts[0]
                        qrs_rpeaks.append(int(float(str_time) * 1000))
        except Exception as e:
            print(f"读取标注文件 {path} 时发生错误: {e}")
            return None # 如果读取失败，也返回None

        return np.array(qrs_rpeaks)

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