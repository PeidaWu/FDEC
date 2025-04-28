import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from onelayer_functional_autoencoder import normalize_data

def visualize_clusters(embeddings, labels, method_name, features=None, t_grid=None):
    """可视化聚类结果，使用曲线图显示不同类别的曲线"""
    # 如果提供了原始特征和时间网格，则绘制曲线图
    if features is not None and t_grid is not None:
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels)
        
        # 为每个类别选择不同的颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制每个类别的曲线
        for i, label in enumerate(unique_labels):
            # 获取当前类别的所有样本
            class_samples = features[labels == label]
            
            # 计算该类别的平均曲线
            mean_curve = np.mean(class_samples, axis=0)
            
            # 绘制平均曲线
            plt.plot(t_grid, mean_curve, color=colors[i], linewidth=2, label=f'Cluster {label}')
            
            # 绘制该类别的所有曲线（透明度较低）
            for sample in class_samples:
                plt.plot(t_grid, sample, color=colors[i], alpha=0.1)
        
        plt.title(f'Clustering Curves - {method_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    # 如果没有提供原始特征，则使用t-SNE可视化嵌入向量
    else:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title(f'Clustering Visualization - {method_name}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

def visualize_reconstruction(X, t_grid, model, input_tensor, n_samples=5):
    """可视化原始曲线和重构曲线的对比"""
    # 数据预处理
    X_normalized, X_mean, X_std = normalize_data(X)
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    
    # 获取重构结果
    with torch.no_grad():
        reconstructed, _ = model(input_tensor)
    
    # 转换为numpy数组
    reconstructed = reconstructed.numpy()
    
    # 随机选择样本进行可视化
    np.random.seed(42)  # 设置随机种子以确保结果可重复
    sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    
    # 创建子图
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3*n_samples))
    
    # 绘制每个样本的原始曲线和重构曲线
    for i, idx in enumerate(sample_indices):
        ax = axes[i] if n_samples > 1 else axes
        ax.plot(t_grid, X_normalized[idx], 'b-', label='Original')
        ax.plot(t_grid, reconstructed[idx], 'r--', label='Reconstructed')
        ax.set_title(f'Sample {idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算MSE
    mse = np.mean((X_normalized - reconstructed) ** 2)
    print(f'Mean Squared Error: {mse:.6f}')
    
    return mse

def load_data(file_path):
    """加载数据并分离特征和标签"""
    data = np.loadtxt(file_path)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels