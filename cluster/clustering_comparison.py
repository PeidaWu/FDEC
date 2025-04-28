import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from functional_autoencoder import BasisGenerator, train_autoencoder, encode_data
from onelayer_fdec import train_fdec
from onelayer_fidec import train_fidec
from one_layer_autoencoder_kmeans import autoencoder_kmeans
from multilayer_autoencoder_kmeans import multilayer_autoencoder_kmeans
from multilayer_fdec import train_multilayer_fdec
from multilayer_fidec import train_multilayer_fidec
from fpca_kmeans import fpca_kmeans
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid
import matplotlib.pyplot as plt

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
        from scipy.optimize import linear_sum_assignment
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
    
    acc = cluster_acc(true_labels, pred_labels)
    return acc, nmi, ari

def visualize_clusters(embeddings, labels, method_name, features=None, t_grid=None):
    """可视化聚类结果，使用t-SNE或曲线图"""
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

def main():
    # 设置随机种子以提高实验的可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 加载数据
    features, true_labels = load_data('D:/course/DataScience/FPCA/Project/fDEC/Dataset/CBF/CBF_TEST.txt')
    n_clusters = len(np.unique(true_labels))
    
    # 设置时间网格点
    t_grid = np.linspace(0, 1, features.shape[1])
    
    # 设置傅里叶基函数参数
    bottleneck_size = 4
    n_basis = bottleneck_size * 2 + 1
    basis_generator = BasisGenerator(n_basis)
    
    # 1. FPCA + K-means
    print('\n1. Running FPCA + K-means...')
    fpca_pred_labels, fpca_embeddings = fpca_kmeans(
        features, t_grid, n_clusters, n_components=bottleneck_size
    )
    fpca_acc, fpca_nmi, fpca_ari = cluster_metrics(true_labels, fpca_pred_labels)
    print(f'FPCA + K-means - ACC: {fpca_acc:.4f}, NMI: {fpca_nmi:.4f}, ARI: {fpca_ari:.4f}')
    visualize_clusters(fpca_embeddings, fpca_pred_labels, 'FPCA + K-means', features, t_grid)
    
    # 2. 单层 Autoencoder + K-means
    print('\n2. Running One-layer Autoencoder + K-means...')
    ae_pred_labels, encoded_data = autoencoder_kmeans(
        features, t_grid, n_clusters, basis_generator,
        hidden_size=32,
        bottleneck_size=4,
        basis_dim=n_basis,
        lr=0.001,
        batch_size=64,
        n_epochs=300
    )
    ae_acc, ae_nmi, ae_ari = cluster_metrics(true_labels, ae_pred_labels)
    print(f'One-layer Autoencoder + K-means - ACC: {ae_acc:.4f}, NMI: {ae_nmi:.4f}, ARI: {ae_ari:.4f}')
    visualize_clusters(encoded_data, ae_pred_labels, 'One-layer Autoencoder + K-means', features, t_grid)
    
    # 3. 多层 Autoencoder + K-means
    print('\n3. Running Multilayer Autoencoder + K-means...')
    ml_ae_pred_labels, ml_encoded_data = multilayer_autoencoder_kmeans(
        features, t_grid, n_clusters, basis_generator,
        hidden_size1=64,
        hidden_size2=16,
        bottleneck_size=4,
        basis_dim=n_basis,
        lr=0.001,
        batch_size=64,
        n_epochs=300
    )
    ml_ae_acc, ml_ae_nmi, ml_ae_ari = cluster_metrics(true_labels, ml_ae_pred_labels)
    print(f'Multilayer Autoencoder + K-means - ACC: {ml_ae_acc:.4f}, NMI: {ml_ae_nmi:.4f}, ARI: {ml_ae_ari:.4f}')
    visualize_clusters(ml_encoded_data, ml_ae_pred_labels, 'Multilayer Autoencoder + K-means', features, t_grid)
    
    # 4. 单层 Functional DEC
    print('\n4. Running One-layer Functional DEC...')
    fdec_model, fdec_pred_labels, fdec_embeddings, fdec_history = train_fdec(
        features, t_grid, basis_generator, n_clusters,
        bottleneck_size=bottleneck_size,
        ae_epochs=300, 
        dec_epochs=30,
        true_labels=true_labels
    )
    fdec_acc, fdec_nmi, fdec_ari = cluster_metrics(true_labels, fdec_pred_labels)
    print(f'One-layer Functional DEC - ACC: {fdec_acc:.4f}, NMI: {fdec_nmi:.4f}, ARI: {fdec_ari:.4f}')
    visualize_clusters(fdec_embeddings, fdec_pred_labels, 'One-layer Functional DEC', features, t_grid)
    
    # 5. 多层 Functional DEC
    print('\n5. Running Multilayer Functional DEC...')
    ml_fdec_model, ml_fdec_pred_labels, ml_fdec_embeddings, ml_fdec_history = train_multilayer_fdec(
        features, t_grid, basis_generator, n_clusters,
        hidden_size1=64,
        hidden_size2=16,
        bottleneck_size=bottleneck_size,
        ae_epochs=300, 
        dec_epochs=30,
        true_labels=true_labels
    )
    ml_fdec_acc, ml_fdec_nmi, ml_fdec_ari = cluster_metrics(true_labels, ml_fdec_pred_labels)
    print(f'Multilayer Functional DEC - ACC: {ml_fdec_acc:.4f}, NMI: {ml_fdec_nmi:.4f}, ARI: {ml_fdec_ari:.4f}')
    visualize_clusters(ml_fdec_embeddings, ml_fdec_pred_labels, 'Multilayer Functional DEC', features, t_grid)
    
    # 6. 单层 Functional IDEC
    print('\n6. Running One-layer Functional IDEC...')
    fidec_model, fidec_pred_labels, fidec_embeddings, fidec_history = train_fidec(
        features, t_grid, basis_generator, n_clusters,
        bottleneck_size=bottleneck_size,
        ae_epochs=300,  
        idec_epochs=30,
        gamma=0.1, 
        true_labels=true_labels
    )  
    fidec_acc, fidec_nmi, fidec_ari = cluster_metrics(true_labels, fidec_pred_labels)
    print(f'One-layer Functional IDEC - ACC: {fidec_acc:.4f}, NMI: {fidec_nmi:.4f}, ARI: {fidec_ari:.4f}')
    visualize_clusters(fidec_embeddings, fidec_pred_labels, 'One-layer Functional IDEC', features, t_grid)

    # 7. 多层 Functional IDEC (0.1)
    print('\n7. Running Multilayer Functional IDEC...')
    ml_fidec_model, ml_fidec_pred_labels, ml_fidec_embeddings,ml_fidec_history = train_multilayer_fidec(
        features, t_grid, basis_generator, n_clusters,
        hidden_size1=64,
        hidden_size2=16,
        bottleneck_size=bottleneck_size,
        ae_epochs=300,  
        idec_epochs=30,
        gamma=0.1, 
        true_labels=true_labels
    )  
    ml_fidec_acc, ml_fidec_nmi, ml_fidec_ari = cluster_metrics(true_labels, ml_fidec_pred_labels)
    print(f'Multilayer Functional IDEC - ACC: {ml_fidec_acc:.4f}, NMI: {ml_fidec_nmi:.4f}, ARI: {ml_fidec_ari:.4f}')
    visualize_clusters(ml_fidec_embeddings, ml_fidec_pred_labels, 'Multilayer Functional IDEC', features, t_grid)

    # 比较所有方法的性能
    print('\n\n性能比较汇总:')
    print(f'FPCA + K-means - ACC: {fpca_acc:.4f}, NMI: {fpca_nmi:.4f}, ARI: {fpca_ari:.4f}')
    print(f'One-layer Autoencoder + K-means - ACC: {ae_acc:.4f}, NMI: {ae_nmi:.4f}, ARI: {ae_ari:.4f}')
    print(f'Multilayer Autoencoder + K-means - ACC: {ml_ae_acc:.4f}, NMI: {ml_ae_nmi:.4f}, ARI: {ml_ae_ari:.4f}')
    print(f'One-layer Functional DEC - ACC: {fdec_acc:.4f}, NMI: {fdec_nmi:.4f}, ARI: {fdec_ari:.4f}')
    print(f'Multilayer Functional DEC - ACC: {ml_fdec_acc:.4f}, NMI: {ml_fdec_nmi:.4f}, ARI: {ml_fdec_ari:.4f}')
    print(f'One-layer Functional IDEC - ACC: {fidec_acc:.4f}, NMI: {fidec_nmi:.4f}, ARI: {fidec_ari:.4f}')
    print(f'Multilayer Functional IDEC - ACC: {ml_fidec_acc:.4f}, NMI: {ml_fidec_nmi:.4f}, ARI: {ml_fidec_ari:.4f}')

    # 绘制比较图
    methods = ['FPCA+KM', 'AE+KM', 'ML-AE+KM', 'FDEC', 'ML-FDEC', 'FIDEC', 'ML-FIDEC']
    acc_values = [fpca_acc, ae_acc, ml_ae_acc, fdec_acc, ml_fdec_acc, fidec_acc, ml_fidec_acc]
    nmi_values = [fpca_nmi, ae_nmi, ml_ae_nmi, fdec_nmi, ml_fdec_nmi, fidec_nmi, ml_fidec_nmi]
    ari_values = [fpca_ari, ae_ari, ml_ae_ari, fdec_ari, ml_fdec_ari, fidec_ari, ml_fidec_ari]
    
    plt.figure(figsize=(15, 6))
    
    x = np.arange(len(methods))
    width = 0.25
    
    plt.bar(x - width, acc_values, width, label='ACC')
    plt.bar(x, nmi_values, width, label='NMI')
    plt.bar(x + width, ari_values, width, label='ARI')
    
    plt.xlabel('Methods')
    plt.ylabel('Scores')
    plt.title('Clustering Performance Comparison')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('clustering_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()