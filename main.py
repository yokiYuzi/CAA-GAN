# main.py

import os
import glob
import re
from parameter import *
from trainer import Trainer
from Tester import Tester  # 导入我们新建的 Tester 类
from data_loader import Data_Loader,Data_Item
from torch.backends import cudnn
from utils import make_folder
from torch.utils.data import DataLoader
from data_loader import FECGDataset

def find_latest_model_step(model_dir):
    """
    在指定目录中查找并返回最新保存模型的迭****。
    通过解析文件名（如 '200000_G_AECG2FECG.pth'）中的数字来实现。
    """
    # 匹配所有生成器模型文件
    list_of_files = glob.glob(os.path.join(model_dir, '*_G_AECG2FECG.pth'))
    
    if not list_of_files:
        return -1 # 如果找不到任何模型文件，返回-1

    latest_step = -1
    
    # 定义一个正则表达式来从文件名中提取前面的数字
    pattern = re.compile(r'(\d+)_G_AECG2FECG.pth')

    for file_path in list_of_files:
        # 获取文件名
        file_name = os.path.basename(file_path)
        # 尝试匹配
        match = pattern.search(file_name)
        if match:
            # 如果匹配成功，提取数字并更新最大值
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                
    return latest_step

def main(config):
    # For fast training
    cudnn.benchmark = True
    
    # 创建所有必要的目录
    model_save_dir = os.path.join(config.model_save_path, config.version)
    make_folder(model_save_dir, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)

    # --- 数据加载 ---
    # 保持您原有的数据加载逻辑不变
    data_item = Data_Item()
    train_dataset = FECGDataset(data_item, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    test_dataset = FECGDataset(data_item, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0)

    # =================================================================================== #
    #                                 1. 训练阶段                                         #
    # =================================================================================== #
    # 不再需要 if/else 判断，直接进入训练流程
    if config.train:
        print("="*20, "开始训练阶段", "="*20)
        if config.model == 'sagan':
            trainer = Trainer(train_dataloader, config)
            trainer.train()
        elif config.model == 'qgan':
            # 如果您有 qgan_trainer，可以在这里实例化
            print("QGAN 训练器尚未实现，跳过训练。")
            pass
        print("="*20, "训练阶段完成", "="*20)
    else:
        print("通过参数跳过了训练阶段，将直接尝试测试。")


    # =================================================================================== #
    #                                 2. 自动评估阶段                                     #
    # =================================================================================== #
    print("\n", "="*20, "开始自动评估阶段", "="*20)
    
    # 自动查找最新的模型迭****
    latest_model_step = find_latest_model_step(model_save_dir)

    if latest_model_step == -1:
        print(f"错误：在目录 {model_save_dir} 中找不到任何兼容的模型文件。")
        print("无法执行评估。请先确保模型已成功训练并保存。")
        return # 退出程序

    print(f"已自动找到最新的模型迭****：{latest_model_step}")
    
    # 将找到的最新迭****设置到 config 中，以便 Tester 使用
    config.pretrained_model = latest_model_step
    
    # 初始化并运行 Tester
    tester = Tester(test_dataloader, config)
    tester.test()
    print("="*20, "自动评估阶段完成", "="*20)


if __name__ == '__main__':
    config = get_parameters()
    main(config)