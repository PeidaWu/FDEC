import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import BSpline
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_data(X):
    """对时间序列数据进行归一化"""
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_normalized = (X - X_mean) / X_std
    return X_normalized, X_mean, X_std

class BasisGenerator:
    """基函数生成器类"""
    def __init__(self, n_basis):
        self.n_basis = n_basis
    
    def generate_basis(self, t):
        """生成傅里叶基函数"""
        basis_matrix = np.zeros((len(t), self.n_basis))
        basis_matrix[:, 0] = 1.0  # 常数项
        
        for i in range(1, self.n_basis):
            if i % 2 == 1:
                # 正弦项
                k = (i + 1) // 2
                basis_matrix[:, i] = np.sin(2 * np.pi * k * t)
            else:
                # 余弦项
                k = i // 2
                basis_matrix[:, i] = np.cos(2 * np.pi * k * t)
        
        return basis_matrix

class FunctionalAutoencoder(nn.Module):
    """函数型自编码器模型"""
    def __init__(self, input_size, hidden_size, bottleneck_size, basis_dim, basis_eval, dropout_rate=0.2):
        super(FunctionalAutoencoder, self).__init__()
        
        # 基函数评估矩阵
        self.basis_eval = torch.tensor(basis_eval, dtype=torch.float32).t()
        
        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, bottleneck_size)
        )
        
        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, input_size * basis_dim)
        )
        
        # 参数
        self.basis_dim = basis_dim
        self.weights = nn.Parameter(torch.zeros((basis_dim, 1)))
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 重塑并应用基函数
        decoded = decoded.reshape(-1, self.basis_dim, self.basis_eval.shape[0])
        basis_output = torch.matmul(decoded, self.basis_eval)
        basis_activation = torch.relu(basis_output)
        
        # 最终重构
        output = torch.sum(basis_activation * self.weights.reshape(1, self.basis_dim, 1), dim=1)
        
        return output, encoded

def train_autoencoder(X, t_grid, basis_generator, hidden_size=32, bottleneck_size=4,
                     basis_dim=9, lr=0.01, batch_size=64, n_epochs=300, seed=42):
    """训练函数型自编码器
    
    参数:
        X: 输入数据 (n_samples, n_points)
        t_grid: 时间网格点
        basis_generator: 基函数生成器实例
        hidden_size: 隐藏层大小
        bottleneck_size: 瓶颈层大小
        basis_dim: 基函数维度
        lr: 学习率
        batch_size: 批次大小
        n_epochs: 训练轮数
        seed: 随机种子
        
    返回:
        model: 训练好的模型
        losses: 每个epoch的损失值列表
    """
    # 设置随机种子
    set_seed(seed)
    
    # 数据预处理
    X_normalized, X_mean, X_std = normalize_data(X)
    
    # 生成基函数矩阵
    basis_matrix = basis_generator.generate_basis(t_grid)
    
    # 计算输入数据在基函数上的投影
    input_data = np.zeros((X.shape[0], basis_matrix.shape[1]))
    for i in range(X.shape[0]):
        input_data[i] = np.sum(X_normalized[i].reshape(-1, 1) * basis_matrix, axis=0)
    
    # 转换为PyTorch张量
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    
    # 初始化模型
    model = FunctionalAutoencoder(
        input_size=basis_matrix.shape[1],
        hidden_size=hidden_size,
        bottleneck_size=bottleneck_size,
        basis_dim=basis_dim,
        basis_eval=basis_matrix
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练循环
    losses = []
    for epoch in range(n_epochs):
        
        total_loss = 0
        for i in range(0, len(input_tensor), batch_size):
            batch_input = input_tensor[i:i + batch_size]
            batch_target = X_tensor[i:i + batch_size]
            
            # 前向传播
            reconstruction, _ = model(batch_input)
            loss = criterion(reconstruction, batch_target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均损失并保存
        avg_loss = total_loss/len(input_tensor)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}')
    
    return model, losses

def encode_data(model, X, t_grid, basis_generator):
    """使用训练好的自编码器对新数据进行编码"""
    # 数据预处理
    X_normalized, _, _ = normalize_data(X)
    
    # 生成基函数矩阵
    basis_matrix = basis_generator.generate_basis(t_grid)
    
    # 计算输入数据在基函数上的投影
    input_data = np.zeros((X.shape[0], basis_matrix.shape[1]))
    for i in range(X.shape[0]):
        input_data[i] = np.sum(X_normalized[i].reshape(-1, 1) * basis_matrix, axis=0)
    
    # 转换为PyTorch张量
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # 编码
    with torch.no_grad():
        _, encoded = model(input_tensor)
    
    return encoded.numpy()