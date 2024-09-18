import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# 设置参数
gamma = 1  # 根据用户要求
tau = 0.01  # 根据用户要求

# 定义 alpha 的取值范围
alpha_values = np.linspace(-3, 1, 500)  # 从 -5 到 5，取 100 个点
spectral_norms = []

# 计算每个 alpha 对应的谱范数
for alpha in alpha_values:
    D = np.array([[0, 1], [alpha, -gamma]])
    D_tau = D * tau
    exp_D_tau = expm(D_tau)
    spectral_norm = np.linalg.norm(exp_D_tau, ord=2)
    spectral_norms.append(spectral_norm)
    if spectral_norm < 1:
        print(alpha)
# 绘制 alpha 对应的谱范数图
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, spectral_norms, label='Spectral Norm of $e^{D\\tau}$')
plt.xlabel('Alpha')
plt.ylabel('Spectral Norm')
plt.title('Spectral Norm vs Alpha')
plt.grid(True)
plt.legend()
plt.show()
