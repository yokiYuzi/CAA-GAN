# Utils/TrainUtils.py

from Utils.DataUtils import DataUtils, DataUtils_NIFECGDB
import numpy as np
from sklearn.model_selection import train_test_split

class TrainUtils():
    def __init__(self):
        self.dataUtils_NIFECGDB = DataUtils_NIFECGDB()
        self.dataUtils = DataUtils()

    # 【核心修正】我们将重写此方法，使其能够处理标注文件缺失的情况
    def prepareData(self, delay=0, test_size=0.2, random_state=42):
        print("正在加载和预处理所有数据...")

        # 初始化用于存储所有有效数据的容器
        all_ecg_data = []
        all_fecg_data = []
        all_fqrs_data = []
        
        # 数据集中的文件索引
        file_indices = [1, 4, 7, 8, 10]
        current_total_length = 0

        # 遍历所有数据文件
        for index in file_indices:
            try:
                ecg_single, fecg_single, fqrs_single = self.dataUtils.readData(index)

                # 【关键检查】如果找不到QRS标注文件，则跳过此数据记录
                if fqrs_single is None:
                    print(f"警告：找不到文件 r{index:02d}.qrs 的标注信息，将跳过此文件。")
                    continue

                # 如果数据有效，则添加到列表中
                all_ecg_data.append(ecg_single)
                all_fecg_data.append(fecg_single)
                # 根据已加载数据的总长度，校正新QRS坐标
                all_fqrs_data.append(fqrs_single + current_total_length)
                
                # 更新当前已加载信号的总长度
                current_total_length += ecg_single.shape[1]

            except Exception as e:
                print(f"加载文件 r{index:02d}.edf 时发生未知错误，将跳过: {e}")
                continue
        
        # 如果没有任何文件被成功加载，则抛出错误
        if not all_ecg_data:
            raise ValueError("严重错误：未能成功加载任何有效的数据和标注。请检查ADFECGDB数据集是否存在且完整（包含.edf和.qrs文件）。")

        # 将所有有效数据拼接成一个大的数组
        ecgAll = np.concatenate(all_ecg_data, axis=1)
        fecg = np.concatenate(all_fecg_data, axis=0)
        fqrs_rpeaks = np.concatenate(all_fqrs_data, axis=0)

        # 后续的窗口化处理现在是完全安全的
        ecgWindows = self.dataUtils.data2window(ecgAll, size=128)
        fecgWindows = self.dataUtils.data2window(fecg, size=128)
        
        fqrs_rpeaks_pro = []
        for i in range(len(ecgWindows)):
            # 这行代码现在不会再因为 fqrs_rpeaks 是 None 而报错
            coo = fqrs_rpeaks - i * 128
            tmp = coo[(coo >= 0) & (coo < 128)]
            fqrs_rpeaks_pro.append(tmp)
        
        fecgWindows = np.expand_dims(fecgWindows, 1)
        fqrs_rpeaks_pro = np.array(fqrs_rpeaks_pro, dtype=object)

        # 使用 train_test_split 进行数据集划分
        print(f"数据加载完成，总样本数: {len(ecgWindows)}")
        print(f"即将按照 8:2 的比例随机分割数据集...")

        X_train, X_test, Y_train, Y_test, fqrs_train, fqrs_test = train_test_split(
            ecgWindows,
            fecgWindows,
            fqrs_rpeaks_pro,
            test_size=test_size,
            random_state=random_state
        )

        print("-" * 30)
        print("数据集分割完成:")
        print(f"训练集样本数: {len(X_train)}")
        print(f"测试集样本数: {len(X_test)}")
        print("-" * 30)
        
        return X_train, X_test, Y_train, Y_test, (fqrs_train, fqrs_test)