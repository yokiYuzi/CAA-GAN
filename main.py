
from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader,Data_Item,FECGDataset
from torch.backends import cudnn
from utils import make_folder


from torch.utils.data import DataLoader
from data_loader import FECGDataset


def main(config):
    
    data_item = Data_Item()
    train_dataset = FECGDataset(data_item.X_train, data_item.Y_train, data_item.fqrs_train)
    val_dataset = FECGDataset(data_item.X_val, data_item.Y_val, data_item.fqrs_val)
    test_dataset = FECGDataset(data_item.X_test, data_item.Y_test, data_item.fqrs_test)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0) # <--- 我们将使用这个


    # 步骤 2: 修改 Trainer 的初始化和调用流程
    if config.train:
        # 将 train, val, test 的 dataloader 都传入 Trainer
        trainer = Trainer(train_dataloader, val_dataloader, test_dataloader, config)
        
        print("Starting training...")
        trainer.train()  # 启动训练和验证流程
        
        print("Training finished. Starting final testing...")
        trainer.test()   # <--- 在训练结束后，调用测试方法
        
    else:
        # 如果您只想运行测试，可以保留或修改这部分逻辑
        # 例如，可以初始化一个trainer，然后直接调用test()
        print("Skipping training. Running test only...")
        trainer = Trainer(train_dataloader, val_dataloader, test_dataloader, config) # 即使不训练，也需要初始化
        trainer.test()

if __name__ == '__main__':
    config = get_parameters()
    main(config)