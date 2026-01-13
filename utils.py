import os
import torch
import random
import numpy as np
import pandas as pd
from moe_model import ModelArgs
from torch.utils.data import Dataset, DataLoader

# 设置随机种子
def set_random_seed(seed):
    """
    设置随机种子，保证每次训练可重复。
    """
    random.seed(seed)  # 训练数据的打乱、随机选择任务
    np.random.seed(seed)  # 随机选择数据、数据集划分
    torch.manual_seed(seed) #  CPU上的随机种子,模型的参数初始化
    torch.cuda.manual_seed(seed) # GPU 上的随机种子,模型的参数初始化
    torch.cuda.manual_seed_all(seed)  # 设置多GPU随机种子，确保多卡训练时的随机性一致
    torch.backends.cudnn.deterministic = True  # 确保每次训练结果一致
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 自动优化，确保每次训练结果一致

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_moe_parameters(model):
    """统计各组件参数量"""
    gate_params = sum(p.numel() for p in model.gate.parameters())
    shared_expert_params = sum(p.numel() for p in model.shared_experts.parameters())
    routed_expert_params = sum(p.numel() for e in model.experts for p in e.parameters())
    output_layer_params = sum(p.numel() for p in model.output_layer.parameters())
    return gate_params, shared_expert_params, routed_expert_params, output_layer_params

def active_parameters(model, top_k, n_routed_experts):
    """计算激活参数量"""
    gate_p, shared_p, routed_p, output_p = count_moe_parameters(model)
    expert_p = routed_p // n_routed_experts
    active_params = gate_p + shared_p + (top_k * expert_p) + output_p
    return active_params

class StandardScaler():
    def __init__(self, args: ModelArgs):
        self.mean = None
        self.std = None
        self.target_column = args.target_column

    def fit(self, data):
        """
        拟合训练数据，计算均值和标准差
        data 是任务的列表，每个任务是一个 Pandas DataFrame
        """
        # 计算每个任务的均值和标准差
        all_data = pd.concat([task.iloc[:, 1] for task in data], axis=0)  # 假设目标列是第二列
        self.mean = all_data.mean()
        self.std = all_data.std()

    def transform(self, data):
        """
        对数据进行标准化
        data 是任务的列表，每个任务是一个 Pandas DataFrame
        """
        normalized_data = []
        task_target_data = data[self.target_column]
        normalized_task_data = (task_target_data - self.mean) / self.std
        # 获取每个任务的 DataFrame
        normalized_task = data.copy()
        normalized_task[self.target_column] = normalized_task_data  # 更新目标列为标准化后的数据
        # 将更新后的任务数据加入列表
        normalized_data.append(normalized_task)
        return normalized_data

    def inverse_transform(self, data):
        """
        对数据进行反标准化
        """
        restored_data = []
        task_data = data[self.target_column] # 目标列数据
        restored_task_data = (task_data * self.std) + self.mean
        restored_task = data.copy()
        restored_task[self.target_column] = restored_task_data  # 更新目标列为反标准化后的数据
        restored_data.append(restored_task)
        return restored_data

def split_data(data, input_window_size, split_ratio=1/2):
    # 检查 data 是否是列表，并且列表中包含一个 DataFrame
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], pd.DataFrame):
            # 如果是单一的 DataFrame
            data = data[0]  # 提取出 DataFrame
        else:
            raise ValueError("Expected a list containing a single DataFrame")
    
    # 确保 data 是一个 DataFrame，进行数据划分
    split_idx = int(len(data) * split_ratio)
    split_idx2 = split_idx - input_window_size
    return [data[:split_idx]], [data[split_idx2:]]

def truncate_tasks_samples(train_tasks, val_tasks, test_tasks, train_samples=None, val_samples=None, test_samples=None):
    
    def _process(tasks, max_samples):
        processed = []
        for df in tasks:  # 直接处理DataFrame
            if max_samples is None:
                processed.append(df)
                continue
            """ 
            if len(df) < max_samples:
                raise ValueError(
                    f"任务数据不足: 需要 {max_samples} 样本，但只有 {len(df)} 个"
                )
            """  
            truncated = df.iloc[:max_samples] # 自动处理 len(df) < max_samples 的情况
            processed.append(truncated)
        return processed

    return (
        _process(train_tasks, train_samples),
        _process(val_tasks, val_samples),
        _process(test_tasks, test_samples)
    )

# 数据标准化
def standardize_data(train_tasks, val_tasks, test_tasks, args: ModelArgs):
    scaler = StandardScaler(args)

    # 对训练集数据拟合并转换
    scaler.fit(train_tasks)

    # 分别对训练集、验证集、测试集进行标准化
    train_tasks_normalized = [scaler.transform(task) for task in train_tasks]
    val_tasks_normalized = [scaler.transform(task) for task in val_tasks]
    test_tasks_normalized = [scaler.transform(task) for task in test_tasks]
    return train_tasks_normalized, val_tasks_normalized, test_tasks_normalized


def sliding_window(data, input_window_size, output_window_size, time_column, target_column):
    """
    滑动窗口生成输入和目标样本
    """
    inputs = []
    targets = []
    
    for i in range(len(data) - input_window_size - output_window_size + 1):
        input_data = data.iloc[i:i + input_window_size][target_column].values  # 只取目标列进行预测
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)  # 转换为 tensor
        
        target_data = data.iloc[i + input_window_size:i + input_window_size + output_window_size][target_column].values
        target_tensor = torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)  # 转换为 tensor
        
        inputs.append(input_tensor)
        targets.append(target_tensor)

    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    
    return inputs, targets


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集类，结合了滑动窗口操作，适合用于训练和测试。
    """
    def __init__(self, data, input_window_size, output_window_size, time_column, target_column):
        self.data = data
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.time_column = time_column
        self.target_column = target_column

    def __len__(self):
        return len(self.data[0]) - self.input_window_size - self.output_window_size + 1

    def __getitem__(self, idx):
        """
        返回数据和目标样本
        """
        task_data = self.data[0]
        input_data = task_data.iloc[idx:idx + self.input_window_size][self.target_column].values
        target_data = task_data.iloc[idx + self.input_window_size: idx + self.input_window_size + self.output_window_size][self.target_column].values
        return torch.tensor(input_data, dtype=torch.float32).unsqueeze(1), torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)


def rmse(outputs, targets):
    return np.sqrt(((outputs - targets) ** 2).mean())

def mae(outputs, targets):
    return np.abs(outputs - targets).mean()

def mape(outputs, targets):
    return (np.abs((outputs - targets) / targets) * 100).mean()


def save_model(model, dataset_name, model_dir="models"):
    model_file = os.path.join(model_dir, f"{dataset_name}_model.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")


def load_model(model, dataset_name, model_dir="models"):
    model_file = os.path.join(model_dir, f"{dataset_name}_model.pth")
    model.load_state_dict(torch.load(model_file))
    print(f"Model loaded from {model_file}")
    return model
