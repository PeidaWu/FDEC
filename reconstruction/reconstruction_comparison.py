import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functional_autoencoder import BasisGenerator, train_autoencoder, encode_data, normalize_data
from multilayer_functional_autoencoder import train_enhanced_autoencoder, encode_data_enhanced, compare_models
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid

# 读取数据
def load_data(file_path):
    data = np.loadtxt(file_path)
    return data

def main():
    # 加载数据
    train_data = load_data('D:/course/DataScience/FPCA/Project/fDEC/Dataset/CBF/CBF_TEST.txt')
    
    # 设置时间网格点
    t_grid = np.linspace(0, 1, train_data.shape[1])
    
    # 设置模型参数
    hidden_size = 50
    hidden_size1 = 80  # 增强版模型第一隐藏层大小
    hidden_size2 = 40  # 增强版模型第二隐藏层大小
    bottleneck_size = 5  # 瓶颈层大小（主成分数量）
    
    # 设置傅里叶基函数参数
    n_basis = bottleneck_size * 2 + 1  # 基函数数量（常数项 + 正弦项 + 余弦项）
    basis_generator = BasisGenerator(n_basis)
    
    basis_dim = bottleneck_size  # 基函数维度与主成分数量一致
    lr = 0.001
    batch_size = 32
    n_epochs = 300
    
    # 训练标准模型
    print('one_layer_autoencoder training...')
    standard_model, standard_losses = train_autoencoder(
        train_data,
        t_grid,
        basis_generator,
        hidden_size=hidden_size,
        bottleneck_size=bottleneck_size,
        basis_dim=basis_dim,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs
    )
    
    # 训练增强版模型
    print('multilayer_autoencoder training...')
    enhanced_model, enhanced_losses = train_enhanced_autoencoder(
        train_data,
        t_grid,
        basis_generator,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        bottleneck_size=bottleneck_size,
        basis_dim=basis_dim,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs
    )
    
    # 绘制损失曲线比较
    print('绘制损失曲线比较...')
    # 计算FPCA的MSE
    fd = FDataGrid(train_data, t_grid)
    fpca = FPCA(n_components=bottleneck_size)
    fpca.fit(fd)
    fpca_recon = fpca.inverse_transform(fpca.transform(fd)).data_matrix[:, :, 0]
    fpca_mse = np.mean((train_data - fpca_recon) ** 2)
    fpca_losses = [fpca_mse] * n_epochs

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), standard_losses, 'b-', label='onelayer Autoencoder')
    plt.plot(range(1, n_epochs + 1), enhanced_losses, 'r-', label='multilayer Autoencoder')
    plt.title('Model Comparison - MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 对训练数据进行编码
    print('对训练数据进行编码...')
    standard_encoded = encode_data(standard_model, train_data, t_grid, basis_generator)
    enhanced_encoded = encode_data_enhanced(enhanced_model, train_data, t_grid, basis_generator)
    print('标准模型编码数据形状:', standard_encoded.shape)
    print('增强模型编码数据形状:', enhanced_encoded.shape)
    
    # 比较模型重构效果
    print('\n比较模型重构效果...')
    standard_mse, enhanced_mse, fpca_mse = compare_models(
        train_data, 
        t_grid, 
        standard_model, 
        enhanced_model, 
        basis_generator, 
        n_components=bottleneck_size
    )
    
    # 打印最终结果总结
    print('\n最终结果总结:')
    print(f'标准模型 MSE: {standard_mse:.6f}')
    print(f'增强模型 MSE: {enhanced_mse:.6f}')
    print(f'FPCA MSE: {fpca_mse:.6f}')

if __name__ == '__main__':
    main()