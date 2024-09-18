import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# 定义参数范围
alpha_values = np.linspace(-3, 1, 500)  # alpha 从 -5 到 5
tau_values = np.linspace(0.01, 0.4, 500)  # tau 从 0.01 到 1
gamma = 1  # 固定 gamma

# 创建网格
alpha_grid, tau_grid = np.meshgrid(alpha_values, tau_values)
spectral_norm_grid = np.zeros_like(alpha_grid)

# 计算每个 (alpha, tau) 对应的谱范数
for i in range(alpha_grid.shape[0]):
    for j in range(alpha_grid.shape[1]):
        alpha = alpha_grid[i, j]
        tau = tau_grid[i, j]
        D = np.array([[0, 1], [alpha, -gamma]])
        D_tau = D * tau
        exp_D_tau = expm(D_tau)
        spectral_norm_grid[i, j] = np.linalg.norm(exp_D_tau, ord=2)

# 绘制三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 创建掩码来标记谱范数小于1的部分
less_than_one_mask = spectral_norm_grid < 1

# 绘制谱范数小于1的部分（绿色）
ax.plot_surface(alpha_grid, tau_grid, np.where(less_than_one_mask, spectral_norm_grid, np.nan), color='green', alpha=0.5, label='Spectral Norm < 1')

# 绘制谱范数大于或等于1的部分（红色）
ax.plot_surface(alpha_grid, tau_grid, np.where(~less_than_one_mask, spectral_norm_grid, np.nan), color='red', alpha=0.5, label='Spectral Norm >= 1')

ax.set_xlabel('Alpha')
ax.set_ylabel('Tau')
ax.set_zlabel('Spectral Norm')
ax.set_title('Spectral Norm vs Alpha and Tau')
plt.show()
