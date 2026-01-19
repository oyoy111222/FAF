import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
import torch
import torch.optim as optim
import json
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from model import MoE, ModelArgs, Shared_Expert, Expert, Gate
from data_loader import *
from utils import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_balance_loss(global_counts, total_selections, n_experts):
    target_count = total_selections / n_experts
    imbalance_loss = ((global_counts - target_count) ** 2).sum()
    return imbalance_loss

def total_loss(predictions, targets, global_counts, total_selections, n_experts, load_balance_lambda=0.1):
    loss = F.mse_loss(predictions, targets.squeeze(-1))
    balance_loss = load_balance_loss(global_counts, total_selections, n_experts)
    return loss + load_balance_lambda * balance_loss
    #return loss

# 解析命令行参数
parser = argparse.ArgumentParser(description="Train a MoE model for time series forecasting")
parser.add_argument('--dataset', type=str, required=True, choices=['ELECTRICI', 'TEMPERA', 'WALMART', 'CO2', 'GDP', 'INDIAN', 'WEATHER'], 
                    help="Dataset to use for training. Choose from 'ELECTRICI', 'TEMPERA', 'WALMART', 'WEATHER', 'INDIAN', 'CO2' or 'GDP'")
args = parser.parse_args()

# 加载配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

# 选择数据集
dataset_name = args.dataset  
dataset_config = config[dataset_name]
seed = dataset_config["seed"] 
set_random_seed(seed)  
inner_steps = dataset_config["inner_steps"]  
lr    = dataset_config["lr"]  
inner_lr    = dataset_config["inner_lr"] 
outer_lr = dataset_config["meta_learning_rate"]
batch_size = dataset_config["batch_size"]      
num_epochs = dataset_config["epochs"]  
patience = dataset_config["early_stopping_patience"] 
load_balance_lambda = dataset_config["load_balance_lambda"]
train_samples = dataset_config.get("train_samples")
val_samples = dataset_config.get("val_samples")
test_samples = dataset_config.get("test_samples")
top_k = dataset_config["n_activated_experts"]
n_routed_experts = dataset_config["n_routed_experts"]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def seed_worker(worker_id):
    worker_seed = dataset_config["seed"] + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

# 加载数据
if dataset_name == "ELECTRICI":
    data = load_electricity_data(dataset_config)
elif dataset_name == "TEMPERA":
    data = load_temperature_data(dataset_config)
elif dataset_name == "WALMART":
    data = load_walmart_data(dataset_config)
elif dataset_name == "CO2":
    data = load_co2emission_data(dataset_config)
elif dataset_name == "WEATHER":
    data = load_weather_data(dataset_config)
elif dataset_name == "INDIAN":
    data = load_indian_data(dataset_config)
else:
    data = load_gdp_data(dataset_config) 

# 获取所有任务的输入数据
task_names = [task[0] for task in data]  
all_data = [task[1] for task in data]  
        
# 按任务划分训练集、验证集和测试集
train_tasks, test_val_tasks = train_test_split(all_data, test_size=0.2, random_state=seed)  # 80% 训练集
val_tasks, test_tasks = train_test_split(test_val_tasks, test_size=0.5, random_state=seed)  # 10% 验证集，10% 测试集
#weather dataset
#train_tasks, test_val_tasks = train_test_split(all_data, test_size=0.4, random_state=seed)  
#val_tasks, test_tasks = train_test_split(test_val_tasks, test_size=0.5, random_state=seed)  

#print(test_tasks)
#print(val_tasks)

print(f"训练集任务个数: {len(train_tasks)}")
print(f"验证集任务个数: {len(val_tasks)}")
print(f"测试集任务个数: {len(test_tasks)}")

# 定义模型
args = ModelArgs(**dataset_config)
model = MoE(args, task_names)
model.apply(init_weights)
model.to(device)

train_tasks, val_tasks, test_tasks = truncate_tasks_samples(train_tasks, val_tasks, test_tasks, train_samples, val_samples, test_samples)

# 对训练集、验证集和测试集进行标准化
train_data, val_data, test_data = standardize_data(train_tasks, val_tasks, test_tasks, args)

# 优化器和损失函数
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# 创建优化器
optimizer_shared = optim.Adam(model.shared_experts.parameters(), lr=outer_lr)
# gate_routed_params = list(model.gate.parameters()) + list(model.experts.parameters())
gate_routed_params = list(model.gate.parameters()) + list(model.experts.parameters()) + list(model.output_layer.parameters())
optimizer_gate_routed = optim.Adam(gate_routed_params, lr=lr)

# 创建 TensorBoard 日志记录器
writer = SummaryWriter()

# 创建早停对象
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def check(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

early_stopping = EarlyStopping(patience)

torch.autograd.set_detect_anomaly(True)
# 训练循环
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_inner_grads = [torch.zeros_like(p) for p in model.shared_experts.parameters()]
    #model.global_counts.zero_()
    #model.total_selections.zero_()
    for task_id, task_data in enumerate(train_data):
        #total_inner_grads = [torch.zeros_like(p) for p in model.shared_experts.parameters()]
        #task_data, _ = split_data(task_data,input_window_size=dataset_config["input_window_size"])
        train_dataset = TimeSeriesDataset(task_data, dataset_config["input_window_size"], dataset_config["output_window_size"], dataset_config["time_column"], dataset_config["target_column"])
        #print(f"len:{len(train_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, worker_init_fn=seed_worker)
        fast_weights = [p.detach().clone().requires_grad_(True) for p in model.shared_experts.parameters()]
        num_batches = len(train_loader)
        #print(train_loader)
        #print(f"num_batches:{num_batches}")
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            for _ in range(inner_steps):
                outputs = model(batch_x, task_id, shared_params=fast_weights)  
                loss_inner = criterion(outputs, batch_y)  
                loss_inner.backward(retain_graph=True)
                with torch.no_grad():
                    for i, w in enumerate(fast_weights):
                        if w.grad is not None:
                            w -= inner_lr * w.grad
                        w.grad = None

            outputs = model(batch_x, task_id, shared_params=fast_weights) 

            gate_weights = model.gate.weight  
            indices = model.gate(batch_x)[1]  
            #loss = criterion(outputs, batch_y)
            batch_counts = torch.bincount(indices.flatten(), minlength=model.n_routed_experts)
            model.global_counts += batch_counts.to(device=model.global_counts.device)
            model.total_selections += indices.numel()  # indices.numel() = batch_size * topk
            loss = total_loss(outputs, batch_y, model.global_counts.detach(), model.total_selections.detach(), model.n_routed_experts, load_balance_lambda)
            loss.backward(retain_graph=True)  

            optimizer_gate_routed.step()
            optimizer_gate_routed.zero_grad()

            for param, fast_param in zip(model.shared_experts.parameters(), fast_weights):
                if fast_param.grad is not None:  
                    if param.grad is None:
                        param.grad = fast_param.grad.detach().clone().to(param.device)
                    else:
                        param.grad += fast_param.grad.detach().clone().to(param.device)

            
            with torch.no_grad():
                for i, p in enumerate(model.shared_experts.parameters()):
                    if p.grad is not None:
                        total_inner_grads[i] += p.grad.detach().clone().cpu().to(device)
                    else:
                        total_inner_grads[i] += torch.zeros_like(total_inner_grads[i]).to(device)
            
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('RMSE/train', rmse(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()), epoch)
            writer.add_scalar('MAE/train', mae(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()), epoch)
            writer.add_scalar('MAPE/train', mape(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()), epoch)

    for i, p in enumerate(model.shared_experts.parameters()):
         p.grad = total_inner_grads[i] / len(train_data)

    optimizer_shared.step()
    optimizer_shared.zero_grad()

    # 保存模型的训练后状态
    trained_model_state_dict = model.state_dict()
    original_global_counts = model.global_counts.clone()
    original_total_selections = model.total_selections.clone()

    # 验证集评估和早停检查  
    model.eval()
    val_loss = 0
    total_inner_grads = [torch.zeros_like(p) for p in model.shared_experts.parameters()]
    for task_id, task_data in enumerate(val_data):
        val_task_data, val_eval_data = split_data(task_data, input_window_size=dataset_config["input_window_size"]) 
        val_dataset = TimeSeriesDataset(val_task_data, dataset_config["input_window_size"], dataset_config["output_window_size"], dataset_config["time_column"], dataset_config["target_column"])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, worker_init_fn=seed_worker)
        fast_weights = [p.detach().clone().requires_grad_(True) for p in model.shared_experts.parameters()]
        num_batches = len(val_loader)
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            #fast_weights = [p.detach().clone().requires_grad_(True) for p in model.shared_experts.parameters()]
            for _ in range(inner_steps):
                outputs = model(batch_x, task_id, shared_params=fast_weights)  
                loss_inner = criterion(outputs, batch_y)
                loss_inner.backward(retain_graph=True)
                with torch.no_grad():
                    for i, w in enumerate(fast_weights):
                        if w.grad is not None:
                            w -= inner_lr * w.grad
                        w.grad = None
            outputs = model(batch_x, task_id, shared_params=fast_weights)  
            gate_weights = model.gate.weight  
            indices = model.gate(batch_x)[1]  
            #loss = criterion(outputs, batch_y)
            batch_counts = torch.bincount(indices.flatten(), minlength=model.n_routed_experts)
            val_global_counts = model.global_counts.detach() + batch_counts.to(device)
            val_total_selections = model.total_selections.detach() + indices.numel()
            loss = total_loss(outputs, batch_y, val_global_counts, val_total_selections,model.n_routed_experts,load_balance_lambda)

            loss.backward(retain_graph=True)  
            optimizer_gate_routed.step()
            optimizer_gate_routed.zero_grad()

            for param, fast_param in zip(model.shared_experts.parameters(), fast_weights):
                if fast_param.grad is not None: 
                    if param.grad is None:
                        param.grad = fast_param.grad.detach().clone().to(param.device)
                    else:
                        param.grad += fast_param.grad.detach().clone().to(param.device)

            with torch.no_grad():
                for i, p in enumerate(model.shared_experts.parameters()):
                    if p.grad is not None:
                        total_inner_grads[i] += p.grad.detach().clone().cpu().to(device)
                    else:
                        total_inner_grads[i] += torch.zeros_like(total_inner_grads[i]).to(device)

        for i, p in enumerate(model.shared_experts.parameters()):
            p.grad = total_inner_grads[i] / num_batches  
        optimizer_shared.step()
        optimizer_shared.zero_grad()

        # 评估阶段
        val_eval_dataset = TimeSeriesDataset(val_eval_data, dataset_config["input_window_size"], dataset_config["output_window_size"], dataset_config["time_column"], dataset_config["target_column"])
        val_eval_loader = DataLoader(val_eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, worker_init_fn=seed_worker)

        with torch.no_grad():
            for batch_x, batch_y in val_eval_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x, task_id)  
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_eval_loader)  

        # 恢复模型参数到训练后的状态
        model.load_state_dict(trained_model_state_dict)
        model.global_counts.copy_(original_global_counts)
        model.total_selections.copy_(original_total_selections)

    # 早停判断
    early_stopping.check(val_loss)
    if early_stopping.stop:
        print("Early stopping triggered")
        break
    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")

# 保存模型的训练后状态
trained_model_state_dict = model.state_dict()
original_global_counts = model.global_counts.clone()
original_total_selections = model.total_selections.clone()

# 测试集评估
model.eval()
test_loss = 0
test_rmse = 0
test_mae = 0
test_mape = 0

for task_id, task_data in enumerate(test_data):
    test_task_data, test_eval_data = split_data(task_data, input_window_size=dataset_config["input_window_size"]) 
    total_inner_grads = [torch.zeros_like(p) for p in model.shared_experts.parameters()]
    test_dataset = TimeSeriesDataset(test_task_data, dataset_config["input_window_size"], dataset_config["output_window_size"], dataset_config["time_column"], dataset_config["target_column"])
    test_loader = DataLoader(test_dataset, batch_size=dataset_config["batch_size"], shuffle=False, drop_last=True, num_workers=0, worker_init_fn=seed_worker)
    fast_weights = [p.detach().clone().requires_grad_(True) for p in model.shared_experts.parameters()]
    num_batches = len(test_loader)
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        #fast_weights = [p.detach().clone().requires_grad_(True) for p in model.shared_experts.parameters()]
        for _ in range(inner_steps):
            outputs = model(batch_x, task_id, shared_params=fast_weights)  
            loss_inner = criterion(outputs, batch_y)
            loss_inner.backward(retain_graph=True)
            with torch.no_grad():
                for i, w in enumerate(fast_weights):
                    if w.grad is not None:
                        w -= inner_lr * w.grad
                    w.grad = None

        outputs = model(batch_x, task_id, shared_params=fast_weights) 

        # 计算损失
        gate_weights = model.gate.weight  
        indices = model.gate(batch_x)[1]  
        #loss = criterion(outputs, batch_y)
        batch_counts = torch.bincount(indices.flatten(), minlength=model.n_routed_experts)
        model.global_counts += batch_counts.to(device=model.global_counts.device)
        model.total_selections += indices.numel()  # indices.numel() = batch_size * topk
        loss = total_loss(outputs, batch_y, model.global_counts.detach(), model.total_selections.detach(), model.n_routed_experts, load_balance_lambda)
    
        loss.backward(retain_graph=True)  
        
        optimizer_gate_routed.step()
        optimizer_gate_routed.zero_grad()

        for param, fast_param in zip(model.shared_experts.parameters(), fast_weights):
            if fast_param.grad is not None:  
                if param.grad is None:
                    param.grad = fast_param.grad.detach().clone().to(param.device)
                else:
                    param.grad += fast_param.grad.detach().clone().to(param.device)

        with torch.no_grad():
            for i, p in enumerate(model.shared_experts.parameters()):
                if p.grad is not None:
                    total_inner_grads[i] += p.grad.detach().clone().cpu().to(device)
                else:
                    total_inner_grads[i] += torch.zeros_like(total_inner_grads[i]).to(device)

    for i, p in enumerate(model.shared_experts.parameters()):
        p.grad = total_inner_grads[i] / num_batches  
    optimizer_shared.step()
    optimizer_shared.zero_grad()
    # 评估阶段
    test_eval_dataset = TimeSeriesDataset(test_eval_data, dataset_config["input_window_size"], dataset_config["output_window_size"], dataset_config["time_column"], dataset_config["target_column"])
    test_eval_loader = DataLoader(test_eval_dataset, batch_size=dataset_config["batch_size"], shuffle=False, drop_last=True, num_workers=0, worker_init_fn=seed_worker)

    with torch.no_grad():
        print(f"batch:{len(test_eval_loader)}")
        for batch_x, batch_y in test_eval_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x, task_id)  
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            print(f"rmse:{rmse(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())}")
            print(f"mae:{mae(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())}")
            print(f"mape:{mape(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())}")
            # 计算指标
            test_rmse += rmse(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
            test_mae += mae(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
            test_mape += mape(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())

    model.load_state_dict(trained_model_state_dict)
    model.global_counts.copy_(original_global_counts)
    model.total_selections.copy_(original_total_selections)
    
print(f"测试任务数:{len(test_data)}")
num_test = len(test_data)  
num_batch = num_test * len(test_eval_loader)
test_loss /= num_batch 
test_rmse /= num_batch
test_mae /= num_batch
test_mape /= num_batch

print(f"Test Loss: {test_loss:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test MAPE: {test_mape:.4f}")

# 保存训练好的模型
model_filename = f"{dataset_name.lower()}"
save_model(model, model_filename)
print(f"Model saved to {model_filename}")
writer.close()

total_params = count_parameters(model)
print(f"Total Parameters: {total_params} ") 

active_params = active_parameters(model, top_k, n_routed_experts)
print(f"Active Parameters: {active_params} ") 
