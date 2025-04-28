import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from multilayer_functional_autoencoder import EnhancedFunctionalAutoencoder, train_enhanced_autoencoder, encode_data_enhanced, set_seed
from onelayer_functional_autoencoder import BasisGenerator, normalize_data

def multilayer_autoencoder_kmeans(features, t_grid, n_clusters, basis_generator, hidden_size1=64, hidden_size2=16,
                      bottleneck_size=4, basis_dim=9, lr=0.001, batch_size=64, n_epochs=300, seed=42):
    """使用多层函数型自编码器降维后进行K-means聚类
    
    参数:
        features: 输入数据 (n_samples, n_points)
        t_grid: 时间网格点
        n_clusters: 聚类数量
        basis_generator: 基函数生成器实例
        hidden_size1: 第一隐藏层大小
        hidden_size2: 第二隐藏层大小
        bottleneck_size: 瓶颈层大小
        basis_dim: 基函数维度
        lr: 学习率
        batch_size: 批次大小
        n_epochs: 训练轮数
    
    返回:
        pred_labels: 预测的聚类标签
        encoded_data: 自编码器编码后的特征
        reconstruction_mse: 重构MSE
    """
    # 设置随机种子
    set_seed(seed)
    
    # 训练多层自编码器
    autoencoder, input_tensor, X_tensor = train_enhanced_autoencoder(
        features, t_grid, basis_generator,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        bottleneck_size=bottleneck_size,
        basis_dim=basis_dim,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=seed
    )
    
    # 使用训练好的自编码器对数据进行编码
    autoencoder.eval()  # 设置为评估模式，禁用dropout
    with torch.no_grad():
        reconstruction, encoded = autoencoder(input_tensor)
    
    encoded_data = encoded.numpy()
    
    # 计算重构MSE
    mse_criterion = torch.nn.MSELoss()
    reconstruction_mse = mse_criterion(reconstruction, X_tensor).item()
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    pred_labels = kmeans.fit_predict(encoded_data)
    
    return pred_labels, encoded_data, reconstruction_mse

def cluster_metrics(true_labels, pred_labels):
    """计算聚类评估指标"""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment
    
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
    
    features, true_labels = load_data('D:/course/DataScience/FPCA/Project/fDEC/Dataset/ECG5000/ECG5000_TEST.txt')
    n_clusters = len(np.unique(true_labels))
    
    # 设置时间网格点
    t_grid = np.linspace(0, 1, features.shape[1])
    
    # 设置傅里叶基函数参数
    bottleneck_size = 4
    n_basis = bottleneck_size * 2 + 1
    basis_generator = BasisGenerator(n_basis)
    
    # 运行多层自编码器+K-means
    print('\nRunning Multilayer Autoencoder + K-means...')
    pred_labels, encoded_data, reconstruction_mse = multilayer_autoencoder_kmeans(
        features, t_grid, n_clusters, basis_generator,
        hidden_size1=64,
        hidden_size2=16,
        bottleneck_size=bottleneck_size,
        basis_dim=n_basis,
        lr=0.001,
        batch_size=64,
        n_epochs=200,
        seed=seed
    )
    
    # 评估聚类结果
    acc, nmi, ari = cluster_metrics(true_labels, pred_labels)
    print(f'Multilayer Autoencoder + K-means - ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}')
    print(f'Reconstruction MSE: {reconstruction_mse:.6f}')
    
    # 可视化聚类结果
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # 使用t-SNE降维可视化
    tsne = TSNE(n_components=2, random_state=seed)  # 保持t-SNE的随机种子固定
    encoded_2d = tsne.fit_transform(encoded_data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1], c=pred_labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Multilayer Autoencoder + K-means Clustering')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

if __name__ == '__main__':
    main()