import numpy as np
import pandas as pd
import torch
import random
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from onelayer_functional_autoencoder import BasisGenerator, train_autoencoder, normalize_data, set_seed

def autoencoder_kmeans(features, t_grid, n_clusters, basis_generator, hidden_size=32,
                      bottleneck_size=4, basis_dim=9, lr=0.001, batch_size=64, n_epochs=300, seed=42):
    """使用函数型自编码器降维后进行K-means聚类
    
    参数:
        features: 输入数据 (n_samples, n_points)
        t_grid: 时间网格点
        n_clusters: 聚类数量
        basis_generator: 基函数生成器实例
        hidden_size: 隐藏层大小
        bottleneck_size: 瓶颈层大小
        basis_dim: 基函数维度
        lr: 学习率
        batch_size: 批次大小
        n_epochs: 训练轮数
        seed: 随机种子
    
    返回:
        pred_labels: 预测的聚类标签
        encoded_data: 自编码器编码后的特征
        losses: 训练过程中的损失值列表
        reconstruction_mse: 重构MSE
    """
    # 设置随机种子
    set_seed(seed)
    
    # 训练自编码器
    autoencoder, losses = train_autoencoder(
        features, t_grid, basis_generator,
        hidden_size=hidden_size,
        bottleneck_size=bottleneck_size,
        basis_dim=basis_dim,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=seed
    )
    
    # 使用训练好的自编码器对数据进行编码
    # 数据预处理
    X_normalized, _, _ = normalize_data(features)
    
    # 生成基函数矩阵
    basis_matrix = basis_generator.generate_basis(t_grid)
    
    # 计算输入数据在基函数上的投影
    input_data = np.zeros((features.shape[0], basis_matrix.shape[1]))
    for i in range(features.shape[0]):
        input_data[i] = np.sum(X_normalized[i].reshape(-1, 1) * basis_matrix, axis=0)
    
    # 转换为PyTorch张量
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # 编码
    autoencoder.eval()  # 设置为评估模式，禁用dropout
    with torch.no_grad():
        _, encoded = autoencoder(input_tensor)
    
    encoded_data = encoded.numpy()
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    pred_labels = kmeans.fit_predict(encoded_data)
    
    # 计算重构MSE
    autoencoder.eval()
    with torch.no_grad():
        reconstruction, _ = autoencoder(input_tensor)
    
    # 计算重构MSE
    mse_criterion = torch.nn.MSELoss()
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    reconstruction_mse = mse_criterion(reconstruction, X_tensor).item()
    
    return pred_labels, encoded_data, losses, reconstruction_mse

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
    # 加载数据
    def load_data(file_path):
        """加载数据并分离特征和标签"""
        data = np.loadtxt(file_path)
        labels = data[:, 0].astype(int)
        features = data[:, 1:]
        return features, labels
    
    # 设置随机种子
    seed = 42
    set_seed(seed)
    
    features, true_labels = load_data('D:/course/DataScience/FPCA/Project/fDEC/Dataset/CBF/CBF_TEST.txt')
    n_clusters = len(np.unique(true_labels))
    
    # 设置时间网格点
    t_grid = np.linspace(0, 1, features.shape[1])
    
    # 设置傅里叶基函数参数
    bottleneck_size = 4
    n_basis = bottleneck_size * 2 + 1
    basis_generator = BasisGenerator(n_basis)
    
    # 运行自编码器+K-means
    print('\nRunning Autoencoder + K-means...')
    pred_labels, encoded_data, losses, reconstruction_mse = autoencoder_kmeans(
        features, t_grid, n_clusters, basis_generator,
        hidden_size=32,
        bottleneck_size=bottleneck_size,
        basis_dim=n_basis,
        lr=0.001,
        batch_size=64,
        n_epochs=300,
        seed=seed
    )
    
    # 打印重构MSE
    print(f'Reconstruction MSE: {reconstruction_mse:.6f}')
    
    # 评估聚类结果
    acc, nmi, ari = cluster_metrics(true_labels, pred_labels)
    print(f'Autoencoder + K-means - ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}')
    
    # 绘制损失随epoch变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.grid(True)

    
    # 可视化聚类结果
    from sklearn.manifold import TSNE
    
    # 使用t-SNE降维可视化
    tsne = TSNE(n_components=2, random_state=seed)
    encoded_2d = tsne.fit_transform(encoded_data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1], c=pred_labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Autoencoder + K-means Clustering')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

if __name__ == '__main__':
    main()