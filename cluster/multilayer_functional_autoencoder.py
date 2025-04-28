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
from onelayer_functional_autoencoder import normalize_data, BasisGenerator

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EnhancedFunctionalAutoencoder(nn.Module):
    """增强版函数型自编码器模型 - 编码器和解码器各增加一层"""
    def __init__(self, input_size, hidden_size1, hidden_size2, bottleneck_size, basis_dim, basis_eval, dropout_rate=0.2):
        super(EnhancedFunctionalAutoencoder, self).__init__()
        
        # 基函数评估矩阵
        self.basis_eval = torch.tensor(basis_eval, dtype=torch.float32).t()
        
        # 编码器层 - 增加一个隐藏层
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size2, bottleneck_size)
        )
        
        # 解码器层 - 增加一个隐藏层
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, input_size * basis_dim)
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
    
    def encode(self, x):
        """仅执行编码过程"""
        return self.encoder(x)

def train_enhanced_autoencoder(X, t_grid, basis_generator, hidden_size1=64, hidden_size2=16, bottleneck_size=4,
                     basis_dim=9, lr=0.01, batch_size=64, n_epochs=300, seed=42):
    """训练增强版函数型自编码器
    
    参数:
        X: 输入数据 (n_samples, n_points)
        t_grid: 时间网格点
        basis_generator: 基函数生成器实例
        hidden_size1: 第一隐藏层大小
        hidden_size2: 第二隐藏层大小
        bottleneck_size: 瓶颈层大小
        basis_dim: 基函数维度
        lr: 学习率
        batch_size: 批次大小
        n_epochs: 训练轮数
        seed: 随机种子
        
    返回:
        model: 训练好的模型
        input_tensor: 输入张量
        X_tensor: 目标张量
    """
    # 设置随机种子以确保结果可重现
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
    model = EnhancedFunctionalAutoencoder(
        input_size=basis_matrix.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        bottleneck_size=bottleneck_size,
        basis_dim=basis_dim,
        basis_eval=basis_matrix
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(n_epochs):
        # 调整学习率
        
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(input_tensor):.6f}')
    
    return model, input_tensor, X_tensor

def encode_data_enhanced(model, X, t_grid, basis_generator, seed=42):
    """使用训练好的增强版自编码器对新数据进行编码
    
    参数:
        model: 训练好的模型
        X: 输入数据
        t_grid: 时间网格点
        basis_generator: 基函数生成器实例
        seed: 随机种子
    """
    # 设置随机种子以确保结果可重现
    set_seed(seed)
    
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