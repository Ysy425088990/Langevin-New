import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad
import time

# 定义参数
gamma = 1
beta = 1  # 假设 beta 为常数
fixed_m = 0.05  # 固定质量参数
num_delta_t = 20  
num_simulations = 5000  # 运行的模拟次数
x_start = -10  # x 轴起始位置
x_end = 10     # x 轴结束位置
num_x_steps = 400  # x 轴的步数
end_t = 10  # 定义不同的停止时间

# 初始化时间步长数组
delta_t_values = np.zeros(num_delta_t)
for i in range(num_delta_t):
    delta_t_values[i] = 0.05 * (i + 1)

# 定义矩阵 D
D2 = np.array([[0, 0], [0, -gamma]])

def matrix_exponential(matrix, t):
    return expm(matrix * t)

# 定义势函数和函数 k(x)
def potential(x):
    return 0.5 * x**2  # 假设势函数为 0.5 * x^2

def k_function(x):
    return 1  # k(x) 为常数

# 计算边缘分布的归一化常数
def normalizing_constant():
    integrand = lambda x: np.exp(-beta * potential(x) + np.log(k_function(x)))
    integral, _ = quad(integrand, -np.inf, np.inf)
    return 1 / integral

# 计算边缘分布
def rho_eq(x):
    Z = normalizing_constant()
    return Z * np.exp(-beta * potential(x) + np.log(k_function(x)))

# 手动计算势函数的梯度
def grad_potential(x):
    return x

# 手动计算 k(x) 的对数梯度
def grad_log_k_function(x):
    return 0

# 计算 x_real
def compute_x_real(x_values):
    return np.array([rho_eq(x) for x in x_values])

# 第二次迭代
def second_iteration(x, p, delta_t, beta):
    grad_potential_x = grad_potential(x)
    grad_log_k_x = grad_log_k_function(x)
    dp = -(grad_potential_x - (1 / beta) * grad_log_k_x) * delta_t
    p_new = p + dp
    x_new = x
    return x_new, p_new

# 使用 D2 进行模拟
def run_simulation_D2(end_t, delta_t):
    D = D2
    h = delta_t / 2
    n_t = int(end_t / delta_t)
    
    final_x_values = []
    start_time = time.time()
    for _ in range(num_simulations):
        # 初始化变量
        x = np.zeros(n_t + 1)
        p = np.zeros(n_t + 1)

        # 使用标准正态分布初始化 x[0]
        x[0] = np.random.normal(0,1)
        p[0] = 0
        
        for i in range(n_t):
            # 第一次迭代使用前向欧拉公式（原来的第二次迭代）
            x_mid, p_mid = second_iteration(x[i], p[i], h, beta)
            
            # 在第一次和第二次迭代之间添加 A 迭代
            x_mid += p_mid * h
            p_mid += 0
            
            # 第二次迭代使用矩阵指数公式（原来的第一次迭代）
            W_h = np.random.normal(0, np.sqrt(delta_t))
            exp_Dh = matrix_exponential(D, delta_t)
            update_vector = exp_Dh @ np.array([x_mid, p_mid]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
            x_mid, p_mid = update_vector
            
            # 在第二次和第三次迭代之间添加 A 迭代
            x_mid += p_mid * h
            p_mid += 0
            
            # 第三次迭代使用前向欧拉公式（原来的第二次迭代）
            x_mid, p_mid = second_iteration(x_mid, p_mid, h, beta)
            
            if i + 1 < len(x):  # 确保不会超出范围
                x[i + 1], p[i + 1] = x_mid, p_mid
            else:
                break
         
        final_x_values.append(x[n_t])
    # 记录运行时间
    end_time = time.time()
    print(f"delta_t = {delta_t:.4f}, Iteration Time = {end_time - start_time:.4f} seconds")
    return np.array(final_x_values)

# 计算 x_real 分布
x_values = np.linspace(x_start, x_end, num_x_steps)
x_real = compute_x_real(x_values)

# 初始化误差数组
errors_D2 = []

# 保存模拟结果以绘图
simulation_results = {}

# 计算并绘制不同时间步长下的误差
for delta_t in delta_t_values:
    # 计算停止时间
     
    
    # 对 D2 矩阵运行模拟
    final_x_values_D2 = run_simulation_D2(end_t, delta_t)
    hist_D2, bins_D2 = np.histogram(final_x_values_D2, bins=50, density=True)  # 使用 density=True 归一化直方图
    bin_centers_D2 = (bins_D2[:-1] + bins_D2[1:]) / 2

    # 计算理论分布在各个 bin 的密度
    theoretical_density = np.interp(bin_centers_D2, x_values, x_real)
    
    # 计算并保存累积密度函数
    cdf_simulated = np.cumsum(hist_D2) * (bins_D2[1] - bins_D2[0])
    cdf_theoretical = np.cumsum(theoretical_density) * (bins_D2[1] - bins_D2[0])
    
    # 归一化累积密度函数
    cdf_simulated /= cdf_simulated[-1]
    cdf_theoretical /= cdf_theoretical[-1]
    
    # 保存模拟结果
    simulation_results[delta_t] = (bin_centers_D2, cdf_simulated, cdf_theoretical)
    
    # 计算误差
    average_cdf_diff_D2 = np.mean(np.abs(cdf_simulated - cdf_theoretical))
    errors_D2.append(average_cdf_diff_D2)

# 绘制每个 delta_t 的累积密度函数 (CDF)
for delta_t in delta_t_values:
    plt.figure()
    bin_centers_D2, cdf_simulated, cdf_theoretical = simulation_results[delta_t]
    
    # 绘制模拟结果和理论分布的CDF
    plt.plot(bin_centers_D2, cdf_simulated, label=f'Simulated CDF (delta_t = {delta_t:.4f})', linestyle='--', color='blue')
    plt.plot(bin_centers_D2, cdf_theoretical, label='Theoretical CDF', linestyle='--', color='red')
    
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title(f'CDF Comparison for delta_t = {delta_t:.4f}')
    plt.legend()
    plt.show()

# 绘制错误与 delta_t 的关系图
plt.figure()
plt.plot(delta_t_values, errors_D2, marker='o', label='Error D2')
plt.xlabel('Time Step (delta_t)')
plt.ylabel('Average CDF Difference (Error)')
plt.title('Error vs. Time Step (delta_t) for D2')
plt.legend()
plt.xscale('log')  # 对数刻度更好地展示范围
plt.show()


