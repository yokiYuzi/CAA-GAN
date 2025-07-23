from Utils.DataUtils import DataUtils, DataUtils_NIFECGDB
import numpy as np
# 【新增】导入 scikit-learn 的数据集划分工具
from sklearn.model_selection import train_test_split


class TrainUtils():
    def __init__(self):
        # 这一部分现在可以安全地执行
        self.dataUtils_NIFECGDB = DataUtils_NIFECGDB()
        self.dataUtils = DataUtils()

    # 【核心修改】我们将重写这个方法以实现功能
    def prepareData(self, delay=0, test_size=0.2, random_state=42):
        
        # 1. 加载并处理全部数据，此部分逻辑不变
        ecgAll, fecg, fqrs_rpeaks = self.dataUtils.readData(1)
        ecgWindows = self.dataUtils.data2window(ecgAll, size=128)
        fecgWindows = self.dataUtils.data2window(fecg, size=128)
        
        fqrs_rpeaks_pro = []
        for i in range(len(ecgWindows)):
            coo = fqrs_rpeaks - i * 128
            tmp = coo[coo >= 0]
            tmp = tmp[tmp < 128]
            fqrs_rpeaks_pro.append(tmp)
        
        fecgWindows = np.expand_dims(fecgWindows, 1)
        # 将 fqrs_rpeaks_pro 转换为numpy数组，并指定dtype=object以处理不等长列表
        fqrs_rpeaks_pro = np.array(fqrs_rpeaks_pro, dtype=object)

        # 2. 【关键】使用 train_test_split 替换旧的切片逻辑
        print(f"数据加载完成，总样本数: {len(ecgWindows)}")
        print(f"即将按照 8:2 的比例随机分割数据集...")

        X_train, X_test, Y_train, Y_test, fqrs_train, fqrs_test = train_test_split(
            ecgWindows,          # 特征 (X)
            fecgWindows,         # 标签 (Y)
            fqrs_rpeaks_pro,     # 与窗口对齐的 fqrs
            test_size=test_size,       # 测试集比例
            random_state=random_state  # 随机种子确保每次分割结果一致
        )

        print("-" * 30)
        print("数据集分割完成:")
        print(f"训练集样本数: {len(X_train)}")
        print(f"测试集样本数: {len(X_test)}")
        print("-" * 30)
        
        # 3. 返回分割好的数据集
        # 为了保持返回5个元素的结构，我们将 fqrs_train 和 fqrs_test 打包成一个元组
        return X_train, X_test, Y_train, Y_test, (fqrs_train, fqrs_test)