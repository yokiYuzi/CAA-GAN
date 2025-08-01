import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import make_folder
from sklearn.metrics import r2_score, mean_squared_error


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class logcosh(nn.Module):
    """
    Log-Cosh 损失函数。
    对于较小的误差，其表现类似于均方误差(MSE)，而对于较大的误差，
    其表现类似于对数误差，使其对异常值不那么敏感。
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, true, pred):
        loss = torch.log(torch.cosh(pred - true))
        return torch.sum(loss)


class LambdaLR():
    """
    用于学习率调度器的辅助类，实现线性衰减。
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "衰减必须在训练结束前开始!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        # 计算学习率的衰减因子
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    """
    权重初始化函数。
    """
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.InstanceNorm1d):
        pass
        
class Trainer(object):
    # 步骤 1: 修改 __init__ 方法以接收所有数据加载器
    def __init__(self, data_loader, val_loader, test_loader, config):
        """
        初始化训练器。
        
        参数:
            data_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
            test_loader: 测试数据加载器。
            config: 包含所有超参数和配置的对象。
        """
        self.data_loader = data_loader
        self.val_loader = val_loader    # <-- 接收并保存验证加载器
        self.test_loader = test_loader  # <-- 接收并保存测试加载器

        # --- 其他所有配置保持不变 ---
        self.model = config.model
        self.adv_loss = config.adv_loss
        
        # 模型超参数
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_AECG_lr = config.g_AECG_lr
        self.g_MECG_lr = config.g_MECG_lr
        self.g_FECG_lr = config.g_FECG_lr
        self.g_BIAS_lr = config.g_BIAS_lr
        self.d_AECG_lr = config.d_AECG_lr
        self.d_MECG_lr = config.d_MECG_lr
        self.d_FECG_lr = config.d_FECG_lr
        self.d_BIAS_lr = config.d_BIAS_lr
        
        self.decay_start_epoch = 1  # 学习率衰减开始的 epoch
        
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        
        # 创建日志、样本和模型保存路径
        self.log_path_base = config.log_path
        self.sample_path_base = config.sample_path
        self.model_save_path_base = config.model_save_path
        self.version = config.version
        
        # 步骤 1: 先调用 make_folder 创建目录，分别传入基础路径和版本号
        make_folder(self.log_path_base, self.version)
        make_folder(self.sample_path_base, self.version)
        make_folder(self.model_save_path_base, self.version)

        # 步骤 2: 然后再定义完整的路径属性供后续使用
        self.log_path = os.path.join(self.log_path_base, self.version)
        self.sample_path = os.path.join(self.sample_path_base, self.version)
        self.model_save_path = os.path.join(self.model_save_path_base, self.version)

        # 构建模型
        self.build_model()
        
        if self.use_tensorboard:
            self.build_tensorboard()

        # 如果指定，加载预训练模型
        if self.pretrained_model:
            self.load_pretrained_model()

    # 步骤 2: 完整重写 train 方法，加入验证逻辑
    def train(self):
        """
        执行完整的训练和验证流程。
        """
        # 确定起始 epoch
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # 初始化计时器和最佳验证损失
        start_time = time.time()
        best_val_loss = float('inf') # 用于记录最佳验证损失

        # 外层循环，代表训练的总轮数 (Epochs)
        for epoch in range(start, self.total_step):
            # --- 训练阶段 ---
            self.model_train() # 切换到训练模式
            train_bar = tqdm(self.data_loader, desc=f'Epoch {epoch+1}/{self.total_step} [Training]')
            
            # 内层循环，遍历一个 epoch 的所有数据批次
            for AECG_signals, FECG_signals, MECG_signals, BIAS_signals in train_bar: 
                # 将数据移动到指定设备
                AECG_signals = AECG_signals.to(device)
                FECG_signals = FECG_signals.to(device)
                MECG_signals = MECG_signals.to(device)
                BIAS_signals = BIAS_signals.to(device)
                
                # 定义真假标签
                valid = torch.ones((AECG_signals.shape[0], 1, 128), dtype=torch.float32).to(device)
                fake = torch.zeros((AECG_signals.shape[0], 1, 128), dtype=torch.float32).to(device)
                
                # --- 生成器(G)训练 ---
                self.optimizer_G_zero_grad()
                
                # === AECG <-> MECG ===
                # 身份损失
                loss_generator_MECG = self.loss_generator(self.G_AECG2MECG(AECG_signals), MECG_signals.float()) * 1
                loss_generator_AECG_from_MECG = self.loss_generator(self.G_MECG2AECG(MECG_signals), AECG_signals.float()) * 1
                # GAN损失
                fake_MECG_signals = self.G_AECG2MECG(AECG_signals)
                loss_forwardGAN_AECG2MECG = self.loss_forwardGAN(self.D_AECG2MECG(fake_MECG_signals), valid)
                fake_AECG_signals_from_MECG = self.G_MECG2AECG(MECG_signals)
                loss_forwardGAN_MECG2AECG = self.loss_forwardGAN(self.D_MECG2AECG(fake_AECG_signals_from_MECG), valid)
                # 循环一致性损失
                loss_cycleGAN_AECG2MECG2AECG = self.loss_cycleGAN(self.G_MECG2AECG(fake_MECG_signals), AECG_signals.float()) * 0.04
                loss_cycleGAN_MECG2AECG2MECG = self.loss_cycleGAN(self.G_AECG2MECG(fake_AECG_signals_from_MECG), MECG_signals.float()) * 0.04
                # 总G损失 (MECG) 并反向传播
                loss_G_total_AECG2MECG = loss_generator_MECG + loss_generator_AECG_from_MECG + loss_forwardGAN_AECG2MECG + loss_forwardGAN_MECG2AECG + loss_cycleGAN_AECG2MECG2AECG + loss_cycleGAN_MECG2AECG2MECG
                loss_G_total_AECG2MECG.backward(retain_graph=True)

                # === AECG <-> FECG ===
                # 身份损失
                loss_generator_FECG = self.loss_generator(self.G_AECG2FECG(AECG_signals), FECG_signals.float()) * 4 # 记录这个损失用于后续比较
                loss_generator_AECG_from_FECG = self.loss_generator(self.G_FECG2AECG(FECG_signals), AECG_signals.float()) * 4
                # GAN损失
                fake_FECG_signals = self.G_AECG2FECG(AECG_signals)
                loss_forwardGAN_AECG2FECG = self.loss_forwardGAN(self.D_AECG2FECG(fake_FECG_signals), valid)
                fake_AECG_signals_from_FECG = self.G_FECG2AECG(FECG_signals)
                loss_forwardGAN_FECG2AECG = self.loss_forwardGAN(self.D_FECG2AECG(fake_AECG_signals_from_FECG), valid)
                # 循环一致性损失
                loss_cycleGAN_AECG2FECG2AECG = self.loss_cycleGAN(self.G_FECG2AECG(fake_FECG_signals), AECG_signals.float()) * 0.04
                loss_cycleGAN_FECG2AECG2FECG = self.loss_cycleGAN(self.G_AECG2FECG(fake_AECG_signals_from_FECG), FECG_signals.float()) * 0.04
                # 总G损失 (FECG) 并反向传播
                loss_G_total_AECG2FECG = loss_generator_FECG + loss_generator_AECG_from_FECG + loss_forwardGAN_AECG2FECG + loss_forwardGAN_FECG2AECG + loss_cycleGAN_AECG2FECG2AECG + loss_cycleGAN_FECG2AECG2FECG
                loss_G_total_AECG2FECG.backward(retain_graph=True)

                # === AECG <-> BIAS ===
                # 身份损失
                loss_generator_BIAS = self.loss_generator(self.G_AECG2BIAS(AECG_signals), BIAS_signals.float()) * 1
                loss_generator_AECG_from_BIAS = self.loss_generator(self.G_BIAS2AECG(BIAS_signals), AECG_signals.float()) * 1
                # GAN损失
                fake_BIAS_signals = self.G_AECG2BIAS(AECG_signals)
                loss_forwardGAN_AECG2BIAS = self.loss_forwardGAN(self.D_AECG2BIAS(fake_BIAS_signals), valid)
                fake_AECG_signals_from_BIAS = self.G_BIAS2AECG(BIAS_signals)
                loss_forwardGAN_BIASAECG = self.loss_forwardGAN(self.D_BIAS2AECG(fake_AECG_signals_from_BIAS), valid)
                # 循环一致性损失
                loss_cycleGAN_AECG2BIAS2AECG = self.loss_cycleGAN(self.G_BIAS2AECG(fake_BIAS_signals), AECG_signals.float()) * 0.04
                loss_cycleGAN_BIAS2AECG2BIAS = self.loss_cycleGAN(self.G_AECG2BIAS(fake_AECG_signals_from_BIAS), BIAS_signals.float()) * 0.04
                # 总G损失 (BIAS) 并反向传播
                loss_G_total_AECG2BIAS = loss_generator_BIAS + loss_generator_AECG_from_BIAS + loss_forwardGAN_AECG2BIAS + loss_forwardGAN_BIASAECG + loss_cycleGAN_AECG2BIAS2AECG + loss_cycleGAN_BIAS2AECG2BIAS
                loss_G_total_AECG2BIAS.backward(retain_graph=True)
                
                # 更新所有生成器的权重
                self.optimizer_G_step()

                # --- 判别器(D)训练 ---
                self.optimizer_D_zero_grad()
                
                # === AECG <-> MECG ===
                loss_D_real_AECG2MECG = self.loss_forwardGAN(self.D_AECG2MECG(AECG_signals), valid)
                loss_D_fake_AECG2MECG = self.loss_forwardGAN(self.D_AECG2MECG(fake_AECG_signals_from_MECG.detach()), fake)
                loss_D_AECG2MECG = (loss_D_real_AECG2MECG + loss_D_fake_AECG2MECG) * 0.5
                loss_D_AECG2MECG.backward(retain_graph=True)

                loss_D_real_MECG2AECG = self.loss_forwardGAN(self.D_MECG2AECG(MECG_signals), valid)
                loss_D_fake_MECG2AECG = self.loss_forwardGAN(self.D_MECG2AECG(fake_MECG_signals.detach()), fake)
                loss_D_MECG2AECG = (loss_D_real_MECG2AECG + loss_D_fake_MECG2AECG) * 0.5
                loss_D_MECG2AECG.backward(retain_graph=True)

                # === AECG <-> FECG ===
                loss_D_real_AECG2FECG = self.loss_forwardGAN(self.D_AECG2FECG(AECG_signals), valid)
                loss_D_fake_AECG2FECG = self.loss_forwardGAN(self.D_AECG2FECG(fake_AECG_signals_from_FECG.detach()), fake)
                loss_D_AECG2FECG = (loss_D_real_AECG2FECG + loss_D_fake_AECG2FECG) * 0.5
                loss_D_AECG2FECG.backward(retain_graph=True)

                loss_D_real_FECG2AECG = self.loss_forwardGAN(self.D_FECG2AECG(FECG_signals), valid)
                loss_D_fake_FECG2AECG = self.loss_forwardGAN(self.D_FECG2AECG(fake_FECG_signals.detach()), fake)
                loss_D_FECG2AECG = (loss_D_real_FECG2AECG + loss_D_fake_FECG2AECG) * 0.5
                loss_D_FECG2AECG.backward(retain_graph=True)
                
                # === AECG <-> BIAS ===
                loss_D_real_AECG2BIAS = self.loss_forwardGAN(self.D_AECG2BIAS(AECG_signals), valid)
                loss_D_fake_AECG2BIAS = self.loss_forwardGAN(self.D_AECG2BIAS(fake_AECG_signals_from_BIAS.detach()), fake)
                loss_D_AECG2BIAS = (loss_D_real_AECG2BIAS + loss_D_fake_AECG2BIAS) * 0.5
                loss_D_AECG2BIAS.backward(retain_graph=True)

                loss_D_real_BIAS_AECG = self.loss_forwardGAN(self.D_BIAS2AECG(BIAS_signals), valid)
                loss_D_fake_BIAS2AECG = self.loss_forwardGAN(self.D_BIAS2AECG(fake_BIAS_signals.detach()), fake)
                loss_D_BIAS2AECG = (loss_D_real_BIAS_AECG + loss_D_fake_BIAS2AECG) * 0.5
                loss_D_BIAS2AECG.backward(retain_graph=True)

                # 更新所有判别器的权重
                self.optimizer_D_step()

            # 在每个 epoch 结束后更新学习率
            self.optimizer_G_lr_step()
            self.optimizer_D_lr_step()
            train_bar.close()

            # --- 验证阶段 ---
            self.model_eval() # 切换到评估模式
            val_loss_total = 0.0
            val_batches = 0
            
            val_bar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.total_step} [Validation]')
            with torch.no_grad(): # 在验证阶段不计算梯度
                for AECG_signals, FECG_signals, _, _ in val_bar:
                    AECG_signals = AECG_signals.to(device)
                    FECG_signals = FECG_signals.to(device)
                    
                    # 只计算我们最关心的生成器损失作为评估指标 (AECG -> FECG)
                    fake_FECG = self.G_AECG2FECG(AECG_signals)
                    val_loss = self.loss_generator(fake_FECG, FECG_signals.float())
                    
                    val_loss_total += val_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0
            
            # --- 日志记录和模型保存 ---
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{self.total_step}] | Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | "
                  f"Train Loss (G_FECG): {loss_generator_FECG.item():.6f} | Validation Loss: {avg_val_loss:.6f}")

            # 步骤 3: 修正模型保存逻辑
            # 根据验证损失来判断是否保存模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"🎉 新的最佳模型! 验证损失降低至 {best_val_loss:.6f}。正在保存模型...")
                # 保存所有需要的模型，使用固定的 "best" 文件名
                torch.save(self.G_AECG2FECG.state_dict(), os.path.join(self.model_save_path, 'best_G_AECG2FECG.pth'))
                # 您也可以在这里保存其他最好的模型
                torch.save(self.G_AECG2MECG.state_dict(), os.path.join(self.model_save_path, 'best_G_AECG2MECG.pth'))
                torch.save(self.G_AECG2BIAS.state_dict(), os.path.join(self.model_save_path, 'best_G_AECG2BIAS.pth'))
                # ... etc.

            # 保存采样图片 (逻辑不变)
            if (epoch + 1) % self.sample_step == 0:
                # 使用验证集中的最后一个批次或重新加载一个批次来生成样本
                fake_FECG_signals = self.G_AECG2FECG(AECG_signals)
                self.sample_images(epoch=epoch, batch_i=epoch, AECG=denorm(AECG_signals.cpu().detach().numpy()), FECG_reconstr=denorm(fake_FECG_signals.cpu().detach().numpy()),FECG=denorm(FECG_signals.cpu().detach().numpy()), sample_path=self.sample_path)


    def test(self):
        """
        在测试集上评估最终模型的性能，并计算 R^2 和 RMSE 指标。
        """
        print("\n" + "="*40)
        print(" " * 12 + "运行最终测试" + " " * 12)
        print("="*40)

        # 步骤 1: 加载训练过程中保存的、表现最好的模型权重
        best_model_path = os.path.join(self.model_save_path, 'best_G_AECG2FECG.pth')
        
        if not os.path.exists(best_model_path):
            print(f"错误: 在 {best_model_path} 未找到最佳模型")
            print("跳过测试阶段。")
            return

        print(f"从以下路径加载最佳模型: {best_model_path}")
        self.G_AECG2FECG.load_state_dict(torch.load(best_model_path))
        
        # 步骤 2: 将所有模型设置为评估模式
        self.model_eval()
        
        # 步骤 3: 准备收集所有真实值和预测值
        all_ground_truths = []
        all_predictions = []
        
        # 步骤 4: 遍历测试数据加载器
        with torch.no_grad():
            for aecg_signals, fecg_signals, _, _ in tqdm(self.test_loader, desc="Testing"):
                aecg_signals = aecg_signals.to(device)
                fecg_signals = fecg_signals.to(device) # 真实目标值

                # 通过生成器获取预测信号
                predicted_fecg_signals = self.G_AECG2FECG(aecg_signals)
                
                # 收集真实值和预测值
                all_ground_truths.append(fecg_signals.cpu().numpy())
                all_predictions.append(predicted_fecg_signals.cpu().numpy())

        # 步骤 5: 将所有批次的数据合并成一个大的Numpy数组
        all_ground_truths = np.concatenate(all_ground_truths, axis=0).reshape(-1, 1)
        all_predictions = np.concatenate(all_predictions, axis=0).reshape(-1, 1)
        
        print(f"测试完成。在 {len(all_ground_truths)} 个数据点上计算指标。")

        # 步骤 6: 计算 R² 和 RMSE
        r2 = r2_score(all_ground_truths, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_ground_truths, all_predictions))
        
        # 步骤 7: 打印结果
        print("\n--- 测试结果 ---")
        print(f"决定系数 (R²) Score: {r2:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print("--------------------\n")

    def build_model(self):
        """
        构建所有的生成器和判别器，并定义损失函数和优化器。
        """
        # --- 创建生成器和判别器 ---
        # AECG <-> MECG
        self.G_AECG2MECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_MECG2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)     
        self.D_AECG2MECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_MECG2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        # AECG <-> FECG
        self.G_AECG2FECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_FECG2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D_AECG2FECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_FECG2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        # AECG <-> BIAS
        self.G_AECG2BIAS = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_BIAS2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D_AECG2BIAS = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_BIAS2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        # --- 定义损失函数 ---
        self.loss_generator = logcosh()      # 用于身份损失
        self.loss_forwardGAN = torch.nn.L1Loss() # 用于GAN损失
        self.loss_cycleGAN = logcosh()       # 用于循环一致性损失
        
        # --- 定义优化器 ---
        self.G_AECG2MECG_optimizer = torch.optim.Adam(self.G_AECG2MECG.parameters(), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_MECG2AECG_optimizer = torch.optim.Adam(self.G_MECG2AECG.parameters(), self.g_MECG_lr, [self.beta1, self.beta2])
        self.D_AECG2MECG_optimizer = torch.optim.Adam(self.D_AECG2MECG.parameters(), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_MECG2AECG_optimizer = torch.optim.Adam(self.D_MECG2AECG.parameters(), self.d_MECG_lr, [self.beta1, self.beta2])
  
        self.G_AECG2FECG_optimizer = torch.optim.Adam(self.G_AECG2FECG.parameters(), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_FECG2AECG_optimizer = torch.optim.Adam(self.G_FECG2AECG.parameters(), self.g_FECG_lr, [self.beta1, self.beta2])
        self.D_AECG2FECG_optimizer = torch.optim.Adam(self.D_AECG2FECG.parameters(), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_FECG2AECG_optimizer = torch.optim.Adam(self.D_FECG2AECG.parameters(), self.d_FECG_lr, [self.beta1, self.beta2])
    
        self.G_AECG2BIAS_optimizer = torch.optim.Adam(self.G_AECG2BIAS.parameters(), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_BIAS2AECG_optimizer = torch.optim.Adam(self.G_BIAS2AECG.parameters(), self.g_BIAS_lr, [self.beta1, self.beta2])
        self.D_AECG2BIAS_optimizer = torch.optim.Adam(self.D_AECG2BIAS.parameters(), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_BIAS2AECG_optimizer = torch.optim.Adam(self.D_BIAS2AECG.parameters(), self.d_BIAS_lr, [self.beta1, self.beta2])

        # --- 定义学习率调度器 ---
        self.G_AECG2MECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2MECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_MECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_MECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2MECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2MECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_MECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_MECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
        self.G_AECG2FECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2FECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_FECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_FECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2FECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2FECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_FECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_FECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
        self.G_AECG2BIAS_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2BIAS_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_BIAS2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_BIAS2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2BIAS_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2BIAS_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_BIAS2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_BIAS2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        # 注意：此函数可能需要根据新的保存逻辑进行调整
        # 目前，它加载的是以 step 命名的旧模型
        model_path = os.path.join(self.model_save_path, f'{self.pretrained_model}_G_AECG2FECG.pth')
        if os.path.exists(model_path):
             self.G_AECG2FECG.load_state_dict(torch.load(model_path))
             print(f'加载预训练模型 (step: {self.pretrained_model})..!'.format(self.pretrained_model))
        else:
            print(f"警告: 找不到预训练模型 {model_path}")

    # --- 优化器辅助函数 ---
    def optimizer_G_zero_grad(self):
        self.G_AECG2MECG_optimizer.zero_grad()
        self.G_MECG2AECG_optimizer.zero_grad()
        self.G_AECG2FECG_optimizer.zero_grad()
        self.G_FECG2AECG_optimizer.zero_grad()
        self.G_AECG2BIAS_optimizer.zero_grad()
        self.G_BIAS2AECG_optimizer.zero_grad()
        
    def optimizer_D_zero_grad(self):    
        self.D_AECG2MECG_optimizer.zero_grad()
        self.D_MECG2AECG_optimizer.zero_grad()
        self.D_AECG2FECG_optimizer.zero_grad()
        self.D_FECG2AECG_optimizer.zero_grad()
        self.D_AECG2BIAS_optimizer.zero_grad()
        self.D_BIAS2AECG_optimizer.zero_grad()
        
    def optimizer_G_step(self):
        self.G_AECG2MECG_optimizer.step()
        self.G_MECG2AECG_optimizer.step()
        self.G_AECG2FECG_optimizer.step()
        self.G_FECG2AECG_optimizer.step()
        self.G_AECG2BIAS_optimizer.step()
        self.G_BIAS2AECG_optimizer.step()
        
    def optimizer_D_step(self):    
        self.D_AECG2MECG_optimizer.step()
        self.D_MECG2AECG_optimizer.step()
        self.D_AECG2FECG_optimizer.step()
        self.D_FECG2AECG_optimizer.step()
        self.D_AECG2BIAS_optimizer.step()
        self.D_BIAS2AECG_optimizer.step()

    def optimizer_G_lr_step(self):     
        self.G_AECG2MECG_exp_lr_scheduler.step()
        self.G_MECG2AECG_exp_lr_scheduler.step()
        self.G_AECG2FECG_exp_lr_scheduler.step()
        self.G_FECG2AECG_exp_lr_scheduler.step()
        self.G_AECG2BIAS_exp_lr_scheduler.step()
        self.G_BIAS2AECG_exp_lr_scheduler.step()

    def optimizer_D_lr_step(self):
        self.D_AECG2MECG_exp_lr_scheduler.step()
        self.D_MECG2AECG_exp_lr_scheduler.step() 
        self.D_AECG2FECG_exp_lr_scheduler.step()
        self.D_FECG2AECG_exp_lr_scheduler.step()
        self.D_AECG2BIAS_exp_lr_scheduler.step()
        self.D_BIAS2AECG_exp_lr_scheduler.step()

    # --- 模型模式切换辅助函数 ---
    def model_train(self):
        """将所有模型切换到训练模式"""
        self.G_AECG2MECG.train()
        self.G_MECG2AECG.train()    
        self.D_AECG2MECG.train()
        self.D_MECG2AECG.train()
        
        self.G_AECG2FECG.train()
        self.G_FECG2AECG.train()
        self.D_AECG2FECG.train()
        self.D_FECG2AECG.train()
    
        self.G_AECG2BIAS.train()
        self.G_BIAS2AECG.train()
        self.D_AECG2BIAS.train()
        self.D_BIAS2AECG.train()

    # 步骤 4: 新增 model_eval 方法
    def model_eval(self):
        """将所有模型切换到评估模式"""
        self.G_AECG2MECG.eval()
        self.G_MECG2AECG.eval()    
        self.D_AECG2MECG.eval()
        self.D_MECG2AECG.eval()
        
        self.G_AECG2FECG.eval()
        self.G_FECG2AECG.eval()
        self.D_AECG2FECG.eval()
        self.D_FECG2AECG.eval()
    
        self.G_AECG2BIAS.eval()
        self.G_BIAS2AECG.eval()
        self.D_AECG2BIAS.eval()
        self.D_BIAS2AECG.eval()

    # --- 样本可视化函数 ---
    def sample_images(self, epoch, batch_i, AECG, FECG_reconstr, FECG, sample_path):
        """
        保存生成的样本图像以供可视化。
        """
        r, c = 1, 3
        # 注意：原始代码中 MECG 标题对应 AECG 信号，这里保持一致
        gen_imgs = [AECG, FECG_reconstr, FECG]
        titles = ['AECG (Input)', 'Reconstructed FECG', 'Ground Truth FECG']
        
        fig, axs = plt.subplots(r, c, figsize=(18, 5))
        for j, (title, data) in enumerate(zip(titles, gen_imgs)):
            # 从批次中选择第一个样本进行绘制
            sample_to_plot = data[0, 0, :]
            axs[j].plot(sample_to_plot)
            axs[j].set_title(title)
            axs[j].set_xlabel("Time Steps")
            axs[j].set_ylabel("Amplitude")
            axs[j].grid(True)
        
        fig.suptitle(f"Epoch {epoch+1}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(sample_path, f"{epoch+1}_{batch_i}.png"), dpi=300)
        plt.close(fig)