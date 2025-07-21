# Tester.py

import os
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sagan_models import Generator
from utils import denorm
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确保 Matplotlib 使用 Agg 后端，避免在无界面的服务器上出错
import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester(object):
    """
    测试器类，用于在训练后自动评估最新/最优模型的性能
    """
    def __init__(self, data_loader, config):
        """
        初始化测试器
        :param data_loader: 测试数据加载器
        :param config: 配置参数，其中 config.pretrained_model 应被设置为要测试的模型迭****
        """
        self.data_loader = data_loader
        self.config = config
        
        # 模型保存路径
        self.model_save_path = os.path.join(config.model_save_path, config.version)
        
        # 结果保存路径
        self.result_path = os.path.join(config.sample_path, config.version, 'test_results')
        os.makedirs(self.result_path, exist_ok=True)
        
        # 构建模型
        self.build_model()

    def build_model(self):
        """
        构建生成器模型并加载指定的预训练权重
        """
        self.G_AECG2FECG = Generator(
            self.config.batch_size,
            self.config.imsize,
            self.config.z_dim,
            self.config.g_conv_dim
        ).to(device)
        
        # config.pretrained_model 应该在调用此类前被主程序设置为最新的模型迭****
        model_path = os.path.join(self.model_save_path, f'{self.config.pretrained_model}_G_AECG2FECG.pth')
        
        if os.path.exists(model_path):
            print(f"测试器：正在从以下路径加载模型: {model_path}")
            self.G_AECG2FECG.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # 这是一个关键的错误检查，防止因路径问题导致程序崩溃
            print(f"错误：在路径 {model_path} 未找到指定的模型文件。")
            print("请检查 model_save_path 和 pretrained_model 参数是否正确。")
            # 退出程序，因为没有模型无法进行测试
            exit()
        
        # 将模型设置为评估模式，这会关闭 Dropout 和 BatchNorm 的训练行为
        self.G_AECG2FECG.eval()

    def test(self):
        """
        执行完整的测试流程，并计算 R² 和 RMSE
        """
        print("开始最终评估...")
        
        all_true_fecg = []
        all_pred_fecg = []

        # torch.no_grad() 上下文管理器可以禁用梯度计算，在评估时能显著提高速度并减少内存占用
        with torch.no_grad():
            pbar = tqdm(self.data_loader, desc="模型评估中")
            for i, (aecg_signals, fecg_signals, _, _) in enumerate(pbar):
                aecg_signals = aecg_signals.to(device)
                fecg_signals = fecg_signals.to(device)

                # 生成预测信号
                pred_fecg_signals = self.G_AECG2FECG(aecg_signals)
                
                # 收集真实信号和预测信号的Numpy数组
                all_true_fecg.append(fecg_signals.cpu().numpy())
                all_pred_fecg.append(pred_fecg_signals.cpu().numpy())

                # 可选：保存一些样本图像进行可视化比较
                if i % self.config.sample_step == 0:
                    self.save_sample_results(i, aecg_signals.cpu().numpy(), fecg_signals.cpu().numpy(), pred_fecg_signals.cpu().numpy())

        # 将所有批次的数据合并成一个大的数组
        all_true_fecg = np.concatenate(all_true_fecg, axis=0)
        all_pred_fecg = np.concatenate(all_pred_fecg, axis=0)

        # 为了计算指标，我们将信号展平为一维向量
        true_flat = all_true_fecg.flatten()
        pred_flat = all_pred_fecg.flatten()

        # 计算评估指标
        print("\n正在计算最终评估指标...")
        rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
        r2 = r2_score(true_flat, pred_flat)

        # 打印格式化的结果
        print("="*40)
        print("         模型性能评估报告")
        print("="*40)
        print(f"均方根误差 (RMSE): {rmse:.6f}")
        print(f"R² (决定系数):    {r2:.6f}")
        print("="*40)
        print(f"评估完成。对比图像已保存至: {self.result_path}")

    def save_sample_results(self, batch_i, aecg, true_fecg, pred_fecg):
        """
        保存单个样本的对比图，用于直观分析
        """
        aecg_sample = aecg[0, 0, :]
        true_fecg_sample = true_fecg[0, 0, :]
        pred_fecg_sample = pred_fecg[0, 0, :]
        
        # --- 将这里的绘图标签改为英文 ---

        # 修改后 ✔️:
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f'Sample {batch_i} - Signal Comparison', fontsize=16)
        
        axs[0].plot(aecg_sample, color='blue', label='Input AECG')
        axs[0].set_title("Input: Abdominal ECG (AECG)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend(loc='upper right')
        
        axs[1].plot(true_fecg_sample, color='green', label='Ground Truth FECG')
        axs[1].plot(pred_fecg_sample, color='red', linestyle='--', label='Generated FECG')
        axs[1].set_title("Comparison: Ground Truth vs. Generated Fetal ECG")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend(loc='upper right')
        
        residual = true_fecg_sample - pred_fecg_sample
        axs[2].plot(residual, color='purple', label='Residual (Truth - Generated)')
        axs[2].set_title("Residual Plot")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("Amplitude Error")
        axs[2].legend(loc='upper right')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.result_path, f'result_comparison_{batch_i}.png')
        fig.savefig(save_path, dpi=200)
        plt.close(fig)