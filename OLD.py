import numpy as np
import matplotlib.pyplot as plt

# 定义潜在函数 f 和权重函数 κ (根据你的需求替换成具体函数)
def f(y):
    return y**2  # 示例：一个简单的平方函数

def kappa(y):
    return np.exp(-y**2)  # 示例：高斯权重

# 初始化参数
eta = 0.01  # 步长参数
phi_max = 1000  # 最大时钟值
k_max = 100000  # 最大迭代次数
y0 = np.random.normal(0, 1)  # 初始值，假设服从标准正态分布
xi = np.random.normal(0, 1, k_max)  # 随机数生成，用于模拟

# 初始化 y 和 φ
y_k = y0
phi_k = 0

samples = []  # 存储样本的列表

# 迭代过程
for k in range(1, k_max + 1):
    # 更新迭代变量 y_k
    grad_f = 2 * y_k  # f(y) = y^2 的梯度
    grad_ln_kappa = -2 * y_k  # ln(κ(y)) 的梯度
    y_k = y_k - eta * grad_f + eta * grad_ln_kappa + np.sqrt(2 * eta) * xi[k-1]
    
    # 更新时钟 φ_k
    phi_k = phi_k + eta / kappa(y_k)
    
    # 收集样本
    if np.floor(phi_k / eta) > np.floor((phi_k - eta / kappa(y_k)) / eta):
        samples.append(y_k)
    
    # 检查时钟是否超过最大值
    if phi_k > phi_max:
        break

# 绘制样本的分布
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')
plt.title('Sample Distribution')
plt.xlabel('y')
plt.ylabel('Density')
plt.show()
