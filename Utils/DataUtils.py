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

        # 准备用于查找标注文件的基本名 (不含扩展名)
        record_name_base = file_name.replace(".edf", "")
        qrs_rpeaks = self.getQRS_wfdb(record_name_base)
        return ecgAll, fecg, qrs_rpeaks

    # 【核心修正】重写 getQRS 方法，使用符号链接技巧和 wfdb 库
    def getQRS_wfdb(self, record_name_base):
        """
        使用符号链接技巧和 wfdb 库来安全地读取非标准命名的二进制 .qrs 文件。
        record_name_base: 记录文件的基本路径和名称 (例如 './ADFECGDB/r01')
        """
        # 1. 定义实际的文件路径和我们需要的标准“假名”路径
        source_ann_path = record_name_base + '.edf.qrs'
        standard_ann_path = record_name_base + '.qrs'
        
        # 检查真正的源文件是否存在
        if not os.path.exists(source_ann_path):
            return None

        link_created = False
        try:
            # 2. 创建一个临时的符号链接，让“假名”指向真正的文件
            if not os.path.exists(standard_ann_path):
                os.symlink(source_ann_path, standard_ann_path)
                link_created = True

            # 3. 现在，wfdb 库可以通过标准名称找到文件了
            annotation = wfdb.rdann(record_name_base, 'qrs')
            qrs_rpeaks = annotation.sample
            return np.array(qrs_rpeaks)

        except Exception as e:
            print(f"使用 wfdb 读取标注文件 {record_name_base} 时发生错误: {e}")
            return None
        finally:
            # 4. 无论成功与否，都清理掉我们创建的临时链接
            if link_created and os.path.exists(standard_ann_path):
                os.remove(standard_ann_path)

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

# 一个空的占位符类，用于解决 TrainUtils.py 中的导入依赖错误
class DataUtils_NIFECGDB():
    def __init__(self):
        pass