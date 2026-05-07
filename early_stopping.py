import os
import torch

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.001, save_dir='./models'):
        self.patience = patience  # 等待改进的轮次数
        self.verbose = verbose  # 如果为True，在每次找到新的最佳模型时打印消息
        self.counter = 0  # 记录没有改进的轮次数
        self.best_score = None  # 最佳验证损失或准确率
        self.early_stop = False  # 停止训练的标志
        self.delta = delta  # 被认为是改进的最小验证损失或准确率的变化量
        self.save_dir = save_dir  # 模型保存目录
        self.best_model_path = None  # 最佳模型路径

    def __call__(self, val_loss):
        if self.best_score is None:
            # 在第一次调用时初始化best_score
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            # 如果验证损失增加，则增加计数器
            self.counter += 1
            # 注释掉验证损失未改进的输出
            # if self.verbose:
            #     print(f'验证损失在 {self.counter} 轮次中没有改进。')
            if self.counter >= self.patience:
                # 如果计数器达到耐心的阈值，则停止训练
                self.early_stop = True
        else:
            # 如果验证损失减少，则重置计数器并更新最佳分数
            self.best_score = val_loss
            self.counter = 0

        return self.early_stop
    
    def save_checkpoint(self, model, epoch, val_loss, val_acc, save_dir=None):
        """保存最佳模型检查点"""
        if save_dir is None:
            save_dir = self.save_dir
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 检查是否需要保存（验证损失是否改进）
        should_save = False
        if self.best_score is None:
            # 第一次保存
            should_save = True
            self.best_score = val_loss
        elif val_loss < self.best_score + self.delta:
            # 验证损失改进
            should_save = True
            self.best_score = val_loss
            self.counter = 0  # 重置计数器
        else:
            # 验证损失未改进
            self.counter += 1
        
        if should_save:
            # 生成模型文件名
            model_filename = f"best_model_epoch_{epoch}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.pth"
            model_path = os.path.join(save_dir, model_filename)
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_score': self.best_score,
            }, model_path)
            
            # 删除之前的模型文件（如果存在）
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path)
                except:
                    pass
            
            self.best_model_path = model_path
            
            if self.verbose:
                if self.best_score is not None:
                    print(f'验证损失改进 ({self.best_score:.6f} --> {val_loss:.6f}). 模型保存到 {model_path}')
                else:
                    print(f'首次保存模型 (验证损失: {val_loss:.6f}). 模型保存到 {model_path}')
        elif self.verbose:
            print(f'验证损失未改进 ({self.best_score:.6f} vs {val_loss:.6f})，跳过模型保存')
    
    def load_best_model(self, model):
        """加载最佳模型"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint
        return None