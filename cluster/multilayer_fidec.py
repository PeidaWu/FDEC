import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import kl_div, softmax, log_softmax, mse_loss
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import random
from onelayer_functional_autoencoder import BasisGenerator, normalize_data, set_seed
from visualization import visualize_clusters, visualize_reconstruction, load_data

class FunctionalAutoencoder(nn.Module):
    """函数型自编码器模型"""
    def __init__(self, input_size, hidden_size1, hidden_size2, bottleneck_size, basis_dim, basis_eval, dropout_rate=0.2):
        super(FunctionalAutoencoder, self).__init__()
        
        # 基函数评估矩阵
        self.basis_eval = torch.tensor(basis_eval, dtype=torch.float32).t()
        
        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size2, bottleneck_size)
        )
        
        # 解码器层
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

class IDEC(nn.Module):
    """改进的深度嵌入聚类模型"""
    def __init__(self, n_clusters, bottleneck_size, alpha=1.0):
        super(IDEC, self).__init__()
        
        # 聚类中心 - 使用固定种子初始化
        torch.manual_seed(42)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, bottleneck_size))
        self.alpha = alpha  # t分布自由度参数
        self.n_clusters = n_clusters
    
    def forward(self, x):
        """计算样本到聚类中心的软分配概率"""
        # 计算样本到每个聚类中心的平方距离
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(x.unsqueeze(1) - self.cluster_centers, 2), dim=2) / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q):
        """计算目标分布"""
        weight = q ** 2 / torch.sum(q, dim=0)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

class FunctionalIDEC(nn.Module):
    """函数型自编码器与IDEC的结合模型"""
    def __init__(self, autoencoder, n_clusters, alpha=1.0):
        super(FunctionalIDEC, self).__init__()
        
        # 预训练的自编码器
        self.autoencoder = autoencoder
        
        # 获取瓶颈层大小
        for layer in self.autoencoder.encoder:
            if isinstance(layer, nn.Linear):
                bottleneck_size = layer.out_features
        
        # IDEC层
        self.idec = IDEC(n_clusters, bottleneck_size, alpha)
    
    def forward(self, x):
        """前向传播"""
        # 获取重构和编码表示
        reconstruction, encoded = self.autoencoder(x)
        
        # 计算软分配概率
        q = self.idec(encoded)
        
        return reconstruction, q, encoded
    
    def target_distribution(self, q):
        """计算目标分布"""
        return self.idec.target_distribution(q)

def train_autoencoder(X, t_grid, basis_generator, hidden_size1=64, hidden_size2=16, bottleneck_size=4,
                     basis_dim=9, lr=0.001, batch_size=64, n_epochs=300, seed=42):
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
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        bottleneck_size=bottleneck_size,
        basis_dim=basis_dim,
        basis_eval=basis_matrix
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 记录训练过程中的损失
    losses = []
    
    # 训练循环
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
        
        # 计算平均损失并记录
        avg_loss = total_loss / len(input_tensor)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}')
    
    return model, input_tensor, X_tensor, losses

def train_fidec(X, t_grid, basis_generator, n_clusters, hidden_size1=64, hidden_size2=16, bottleneck_size=4,
              basis_dim=9, ae_lr=0.001, idec_lr=0.001, batch_size=64, 
              ae_epochs=300, idec_epochs=30, update_interval=3, tol=0.001, 
              gamma=0.1, true_labels=None, pretrained_autoencoder=None, seed=42):
    """训练函数型自编码器与IDEC的结合模型
    
    参数:
        X: 输入数据 (n_samples, n_points)
        t_grid: 时间网格点
        basis_generator: 基函数生成器实例
        n_clusters: 聚类数量
        hidden_size: 隐藏层大小
        bottleneck_size: 瓶颈层大小
        basis_dim: 基函数维度
        ae_lr: 自编码器学习率
        idec_lr: IDEC学习率
        batch_size: 批次大小
        ae_epochs: 自编码器训练轮数
        idec_epochs: IDEC训练轮数
        update_interval: 目标分布更新间隔
        gamma: KL散度损失的权重系数
        true_labels: 真实标签（如果有）
        seed: 随机种子
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
    
    # 使用预训练的自编码器或训练新的自编码器
    if pretrained_autoencoder is not None:
        print("Using pretrained autoencoder...")
        autoencoder = pretrained_autoencoder
        ae_losses = None
    elif ae_epochs > 0:
        print("Step 1: Pretraining autoencoder...")
        autoencoder, input_tensor, X_tensor, ae_losses = train_autoencoder(
            X, t_grid, basis_generator,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            bottleneck_size=bottleneck_size,
            basis_dim=basis_dim,
            lr=ae_lr,
            batch_size=batch_size,
            n_epochs=ae_epochs,
            seed=seed
        )
    else:
        # 如果不训练自编码器，则使用随机初始化的自编码器
        basis_matrix = basis_generator.generate_basis(t_grid)
        autoencoder = FunctionalAutoencoder(
            input_size=basis_matrix.shape[1],
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            bottleneck_size=bottleneck_size,
            basis_dim=basis_dim,
            basis_eval=basis_matrix
        )
        ae_losses = None
    
    # 2. 初始化聚类中心
    print("Step 2: Initializing cluster centers...")
    with torch.no_grad():
        _, features = autoencoder(input_tensor)
        features_np = features.numpy()
    
    # 使用K-means初始化聚类中心
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    y_pred = kmeans.fit_predict(features_np)
    
    # 3. 初始化IDEC模型
    fidec = FunctionalIDEC(autoencoder, n_clusters)
    
    # 使用K-means的聚类中心初始化IDEC的聚类中心
    fidec.idec.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    # 4. 训练IDEC模型
    print("Step 3: Training IDEC model...")
    optimizer = optim.Adam(fidec.parameters(), lr=idec_lr)
    
    # 用于检测收敛的变量
    prev_loss = float('inf')
    converge_count = 0
    required_converge_count = 3  # 连续3次满足收敛条件才停止
    
    # 记录每个epoch的ACC值和损失值
    acc_history = []
    epoch_history = []
    kl_loss_history = []  # 记录KL散度损失
    reconstr_loss_history = []  # 记录重构损失
    total_loss_history = []  # 记录总损失（KL散度损失 + gamma * 重构损失）
    
    for epoch in range(idec_epochs):
        # 计算当前的软分配矩阵
        fidec.eval()
        with torch.no_grad():
            _, q, _ = fidec(input_tensor)
            p = fidec.target_distribution(q)
        
        # 训练模式
        fidec.train()
        
        total_loss = 0
        for i in range(0, len(input_tensor), batch_size):
            batch_input = input_tensor[i:i + batch_size]
            batch_target = X_tensor[i:i + batch_size]
            batch_p = p[i:i + batch_size]
            
            # 前向传播
            batch_x_bar, batch_q, _ = fidec(batch_input)
            
            # 计算重构损失
            reconstr_loss = mse_loss(batch_x_bar, batch_target)
            
            # 计算KL散度损失
            kl_loss = kl_div(batch_q.log(), batch_p, reduction='batchmean')
            
            # 组合损失 - IDEC的关键改进：同时优化重构误差和聚类目标
            loss = kl_loss + gamma * reconstr_loss
            
            # 记录各个批次的损失值，用于后续计算平均损失
            batch_kl_loss = kl_loss.item()
            batch_reconstr_loss = reconstr_loss.item()
            batch_total_loss = loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(input_tensor)
        
        # 计算并记录各类损失的平均值
        fidec.eval()
        with torch.no_grad():
            x_bar, q, _ = fidec(input_tensor)
            p = fidec.target_distribution(q)
            epoch_kl_loss = kl_div(q.log(), p, reduction='batchmean').item()
            epoch_reconstr_loss = mse_loss(x_bar, X_tensor).item()
            epoch_total_loss = epoch_kl_loss + gamma * epoch_reconstr_loss
            
            kl_loss_history.append(epoch_kl_loss)
            reconstr_loss_history.append(epoch_reconstr_loss)
            total_loss_history.append(epoch_total_loss)
        
        # 检查收敛条件 - 相对变化小于0.1%
        if prev_loss != float('inf'):
            rel_change = abs(avg_loss - prev_loss) / prev_loss
            if rel_change < tol:
                converge_count += 1
                if converge_count >= required_converge_count:
                    print(f'Converged at epoch {epoch+1} with loss change < {tol*420}%')
                    break
            else:
                converge_count = 0
        
        prev_loss = avg_loss
        
        # 每个epoch计算当前聚类结果的ACC
        fidec.eval()
        with torch.no_grad():
            _, q, _ = fidec(input_tensor)
            p = fidec.target_distribution(q)
            current_pred = torch.argmax(q, dim=1).numpy()
        
        # 如果提供了真实标签，计算并记录当前ACC
        if true_labels is not None:
            current_acc, current_nmi, current_ari = cluster_metrics(true_labels, current_pred)
            acc_history.append(current_acc)
            epoch_history.append(epoch+1)
            print(f'Epoch [{epoch+1}/{idec_epochs}], Loss: {avg_loss:.6f}, ACC: {current_acc:.4f}, NMI: {current_nmi:.4f}, ARI: {current_ari:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{idec_epochs}], Loss: {avg_loss:.6f}')
            
        # 更新目标分布
        if epoch % update_interval == 0:
            # 获取当前聚类结果用于更新目标分布
            fidec.eval()
            with torch.no_grad():
                _, q, _ = fidec(input_tensor)
                p = fidec.target_distribution(q)
    
    # 5. 获取最终聚类结果
    fidec.eval()
    with torch.no_grad():
        _, q, features = fidec(input_tensor)
        final_pred = torch.argmax(q, dim=1).numpy()
    
    # 如果提供了真实标签，绘制ACC随epoch变化的曲线图
    if true_labels is not None and len(acc_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_history, acc_history, 'b-', marker='o')
        plt.title('Clustering Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.grid(True)
        plt.show()
    
    # 绘制损失随epoch变化的曲线图
    if len(epoch_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_history, kl_loss_history, 'r-', marker='o', label='KL Loss')
        plt.plot(epoch_history, reconstr_loss_history, 'g-', marker='s', label='Reconstruction Loss')
        plt.plot(epoch_history, total_loss_history, 'b-', marker='^', label=f'Total Loss (KL + {gamma} * Reconstr)')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return fidec, final_pred, features.numpy(), ae_losses, (epoch_history, acc_history, kl_loss_history, reconstr_loss_history, total_loss_history) if true_labels is not None else (epoch_history, kl_loss_history, reconstr_loss_history, total_loss_history)

def load_data(file_path):
    """加载数据并分离特征和标签"""
    data = np.loadtxt(file_path)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels

def cluster_metrics(true_labels, pred_labels):
    """计算聚类评估指标"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    # 计算聚类准确率
    def cluster_acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
    
    acc = cluster_acc(true_labels, pred_labels)
    return acc, nmi, ari

def main():    
    # 设置随机种子
    seed = 3
    set_seed(seed)
    
    features, true_labels = load_data('D:/course/DataScience/FPCA/Project/fDEC/Dataset/CBF/CBF_TEST.txt')
    n_clusters = len(np.unique(true_labels))
    
    # 设置时间网格点
    t_grid = np.linspace(0, 1, features.shape[1])
    
    # 设置傅里叶基函数参数
    bottleneck_size = 4
    n_basis = bottleneck_size * 2 + 1
    basis_generator = BasisGenerator(n_basis)
    
    # 训练单层函数型IDEC模型
    print('\nRunning multilayer-layer Functional IDEC...')
    fidec_model, fidec_pred_labels, fidec_embeddings, ae_losses, fidec_history = train_fidec(
        features, t_grid, basis_generator, n_clusters,
        hidden_size1=64,
        hidden_size2=16,
        bottleneck_size=bottleneck_size,
        basis_dim=n_basis,
        ae_epochs=300,
        idec_epochs=30,
        gamma=0.1,
        true_labels=true_labels,
        seed=seed
    )
    
    # 可视化自编码器训练过程中的损失变化
    if ae_losses is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(ae_losses) + 1), ae_losses, 'b-')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.show()
    
    # 可视化原始曲线和重构曲线的对比
    # 获取自编码器
    autoencoder = fidec_model.autoencoder
    
    # 数据预处理
    X_normalized, X_mean, X_std = normalize_data(features)
    
    # 生成基函数矩阵
    basis_matrix = basis_generator.generate_basis(t_grid)
    
    # 计算输入数据在基函数上的投影
    input_data = np.zeros((features.shape[0], basis_matrix.shape[1]))
    for i in range(features.shape[0]):
        input_data[i] = np.sum(X_normalized[i].reshape(-1, 1) * basis_matrix, axis=0)
    
    # 转换为PyTorch张量
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # 可视化重构结果
    mse = visualize_reconstruction(features, t_grid, autoencoder, input_tensor, n_samples=5)
    
    # 可视化聚类结果
    visualize_clusters(fidec_embeddings, fidec_pred_labels, 'One-layer Functional IDEC', features, t_grid)
    
    # 评估聚类结果
    fidec_acc, fidec_nmi, fidec_ari = cluster_metrics(true_labels, fidec_pred_labels)
    print(f'One-layer Functional IDEC - ACC: {fidec_acc:.4f}, NMI: {fidec_nmi:.4f}, ARI: {fidec_ari:.4f}')

if __name__ == "__main__":
    main()