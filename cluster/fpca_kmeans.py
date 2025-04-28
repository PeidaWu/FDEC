import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# 设置随机种子以确保结果可重复
np.random.seed(42)

def fpca_kmeans(features, t_grid, n_clusters, n_components):
    """使用FPCA降维后进行K-means聚类
    
    参数:
        features: 输入数据 (n_samples, n_points)
        t_grid: 时间网格点
        n_clusters: 聚类数量
        n_components: FPCA主成分数量
    
    返回:
        pred_labels: 预测的聚类标签
        fpca_scores: FPCA降维后的特征
        reconstruction_mse: 重构均方误差
    """
    # 将数据转换为FDataGrid对象
    fd = FDataGrid(features, t_grid)
    
    # 使用FPCA进行降维
    fpca = FPCA(n_components=n_components)
    fpca_scores = fpca.fit_transform(fd)
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(fpca_scores)
    
    # 计算重构均方误差
    reconstruction = fpca.inverse_transform(fpca_scores)
    reconstruction_mse = np.mean((features - reconstruction.data_matrix.squeeze())**2)
    
    return pred_labels, fpca_scores, reconstruction_mse

def cluster_acc(y_true, y_pred):
    """计算聚类准确率"""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

def main():
    # 加载数据
    data = np.loadtxt('D:/course/DataScience/FPCA/Project/fDEC/Dataset/Beef/Beef_TEST.txt')
    true_labels = data[:, 0].astype(int)
    features = data[:, 1:]
    
    # 设置参数
    n_clusters = len(np.unique(true_labels))
    t_grid = np.linspace(0, 1, features.shape[1])
    n_components = 4
    
    # 运行FPCA + K-means
    pred_labels, fpca_scores, reconstruction_mse = fpca_kmeans(
        features, t_grid, n_clusters, n_components
    )
    
    # 计算聚类准确率
    acc = cluster_acc(true_labels, pred_labels)
    
    # 计算NMI和ARI指标
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    # 输出结果
    print(f'FPCA + K-means - ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, MSE: {reconstruction_mse:.4f}')

if __name__ == '__main__':
    main()
