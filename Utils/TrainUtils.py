# Utils/TrainUtils.py

from Utils.DataUtils import DataUtils, DataUtils_NIFECGDB
import numpy as np
from sklearn.model_selection import train_test_split

class TrainUtils():
    def __init__(self):
        self.dataUtils_NIFECGDB = DataUtils_NIFECGDB()
        self.dataUtils = DataUtils()

    def prepareData(self, delay=0, test_size=0.2, random_state=42):
        print("正在加载和预处理所有数据...")
        ecgAll, fecg, fqrs_rpeaks = self.dataUtils.readData(1)
        for i in range(2, 6): # 加载所有5个数据文件 (r01, r04, r07, r08, r10)
            try:
                # 注意：数据集的文件名不是连续的，我们按RECORDS文件中的顺序来
                file_indices = [1, 4, 7, 8, 10]
                ecg, fecg_single, fqrs_rpeaks1 = self.dataUtils.readData(file_indices[i-1])
                
                # 更新fqrs的坐标
                fqrs_rpeaks1 = fqrs_rpeaks1 + ecgAll.shape[1]
                fqrs_rpeaks = np.append(fqrs_rpeaks, fqrs_rpeaks1)
                
                ecgAll = np.append(ecgAll, ecg, axis=1)
                fecg = np.append(fecg, fecg_single, axis=0)
            except Exception as e:
                # 如果文件不存在则跳过
                print(f"跳过加载文件 r{i}.edf: {e}")
                continue

        ecgWindows = self.dataUtils.data2window(ecgAll, size=128)
        fecgWindows = self.dataUtils.data2window(fecg, size=128)
        
        fqrs_rpeaks_pro = []
        for i in range(len(ecgWindows)):
            coo = fqrs_rpeaks - i * 128
            tmp = coo[(coo >= 0) & (coo < 128)]
            fqrs_rpeaks_pro.append(tmp)
        
        fecgWindows = np.expand_dims(fecgWindows, 1)
        fqrs_rpeaks_pro = np.array(fqrs_rpeaks_pro, dtype=object)

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