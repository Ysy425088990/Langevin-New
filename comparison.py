import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad
import time
from delta_t import run_simulation_D1,run_simulation_D2,run_simulation_D3
from adapt_deltat import run_simulation_D1_transformed,run_simulation_D2_transformed,run_simulation_D3_transformed
# 定义参数
gamma = 1
beta = 1  # 假设 beta 为常数
num_delta_t = 10
num_simulations = 1000  # 运行的模拟次数
x_start = -10  # x 轴起始位置
x_end = 10     # x 轴结束位置
num_x_steps = 4000  # x 轴的步数
T = 100
end_t = 0
sample_number = 20

# 定义不同的时间步长
fixed_dts = np.zeros(num_delta_t)
for i in range(num_delta_t):
    fixed_dts[i] = 1 * (i + 1)
    
# 动态时间步长函数
def dynamic_time_step(x_prev,fixed_dt):
    return fixed_dt * k_function(x_prev) # 可以根据需要调整

# 定义矩阵 D1 和 D2
D1 = np.array([[0, 1], [0, -gamma]])
D2 = np.array([[0, 0], [0, -gamma]])
D3 = np.array([[0, 1], [0, -gamma]])

def matrix_exponential(matrix, t):
    return expm(matrix * t)

# 定义势函数和函数 k(x)
def potential(x):
    return  0.5 * x ** 2 

def k_function(x):
    return np.exp(-0.1 * x ** 2) # k(x) 为常数

# 计算边缘分布的归一化常数
def normalizing_constant():
    integrand = lambda x: np.exp(-beta * potential(x) )
    integral, _ = quad(integrand, -np.inf, np.inf)
    return 1 / integral

# 计算边缘分布
def rho_eq(x):
    Z = normalizing_constant()
    return Z * np.exp(-beta * potential(x))

# 手动计算势函数的梯度
def grad_potential(x):
    return   x

# 手动计算 k(x) 的对数梯度
def grad_log_k_function(x):
    return  - 0.2 * x 

# 计算 x_real
def compute_x_real(x_values):
    return np.array([rho_eq(x) for x in x_values])

# 第二次迭代
def second_iteration_transformed(x, p, delta_t, beta):
    grad_potential_x = grad_potential(x)
    grad_log_k_x = grad_log_k_function(x)
    dp = -(grad_potential_x - (1 / beta) * grad_log_k_x) * delta_t
    p_new = p + dp
    x_new = x
    return x_new, p_new

# 第二次迭代
def second_iteration(x, p, delta_t, beta):
    grad_potential_x = grad_potential(x)
    dp = -(grad_potential_x) * delta_t
    p_new = p + dp
    x_new = x
    return x_new, p_new


# 计算 x_real 分布
x_values = np.linspace(x_start, x_end, num_x_steps)
x_real = compute_x_real(x_values)

# 初始化错误列表
errors_D1 = []
errors_D2 = []
errors_D3 = []
errors_D1_transformed = []
errors_D2_transformed = []
errors_D3_transformed = []

for fixed_dt in fixed_dts:
    k_max = np.int(T/fixed_dt)

    start_time = time.time()
    # D1 和 D1_transformed 的模拟
    final_x_values_D1 = []
    final_x_values_D1_transformed = []
    final_x_values_D2 = []
    final_x_values_D2_transformed = []
    final_x_values_D3 = []
    final_x_values_D3_transformed = []
    
    
    
    for _ in range(num_simulations):
        initial = [0,0]
        # 运行模拟并保留最后20个数据点
        final_x_values_D1.append(run_simulation_D1(fixed_dt, T, gamma, beta, second_iteration, D1, initial)[-sample_number:])
        final_x_values_D1_transformed.append(run_simulation_D1_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D1,  initial,k_max,k_function)[0])
        final_x_values_D2.append(run_simulation_D2(fixed_dt, T, gamma, beta, second_iteration, D2, initial)[-sample_number:])
        final_x_values_D2_transformed.append(run_simulation_D2_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D2,  initial,k_max,k_function)[0])
        final_x_values_D3.append(run_simulation_D3(fixed_dt, T, gamma, beta, second_iteration, D3, initial)[-sample_number:])
        final_x_values_D3_transformed.append(run_simulation_D3_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D3,  initial,k_max,k_function)[0])
    
    final_x_values_D1 = np.array(final_x_values_D1).flatten()
    final_x_values_D1_transformed = np.array(final_x_values_D1_transformed).flatten()

    hist_D1, bins_D1 = np.histogram(final_x_values_D1, bins=50, density=True)
    bin_centers_D1 = (bins_D1[:-1] + bins_D1[1:]) / 2
    theoretical_density_D1 = np.interp(bin_centers_D1, x_values, x_real)
    cdf_simulated_D1 = np.cumsum(hist_D1) * (bins_D1[1] - bins_D1[0])
    cdf_theoretical_D1 = np.cumsum(theoretical_density_D1) * (bins_D1[1] - bins_D1[0])
    cdf_simulated_D1 /= cdf_simulated_D1[-1]
    cdf_theoretical_D1 /= cdf_theoretical_D1[-1]
    average_cdf_diff_D1 = np.mean(np.abs(cdf_simulated_D1 - cdf_theoretical_D1))
    errors_D1.append(average_cdf_diff_D1)

    hist_D1_transformed, bins_D1_transformed = np.histogram(final_x_values_D1_transformed, bins=50, density=True)
    bin_centers_D1_transformed = (bins_D1_transformed[:-1] + bins_D1_transformed[1:]) / 2
    theoretical_density_D1_transformed = np.interp(bin_centers_D1_transformed, x_values, x_real)
    cdf_simulated_D1_transformed = np.cumsum(hist_D1_transformed) * (bins_D1_transformed[1] - bins_D1_transformed[0])
    cdf_theoretical_D1_transformed = np.cumsum(theoretical_density_D1_transformed) * (bins_D1_transformed[1] - bins_D1_transformed[0])
    cdf_simulated_D1_transformed /= cdf_simulated_D1_transformed[-1]
    cdf_theoretical_D1_transformed /= cdf_theoretical_D1_transformed[-1]
    average_cdf_diff_D1_transformed = np.mean(np.abs(cdf_simulated_D1_transformed - cdf_theoretical_D1_transformed))
    errors_D1_transformed.append(average_cdf_diff_D1_transformed)

    # D2 和 D2_transformed 的模拟
    
    final_x_values_D2 = np.array(final_x_values_D2).flatten()
    final_x_values_D2_transformed = np.array(final_x_values_D2_transformed).flatten()

    hist_D2, bins_D2 = np.histogram(final_x_values_D2, bins=50, density=True)
    bin_centers_D2 = (bins_D2[:-1] + bins_D2[1:]) / 2
    theoretical_density_D2 = np.interp(bin_centers_D2, x_values, x_real)
    cdf_simulated_D2 = np.cumsum(hist_D2) * (bins_D2[1] - bins_D2[0])
    cdf_theoretical_D2 = np.cumsum(theoretical_density_D2) * (bins_D2[1] - bins_D2[0])
    cdf_simulated_D2 /= cdf_simulated_D2[-1]
    cdf_theoretical_D2 /= cdf_theoretical_D2[-1]
    average_cdf_diff_D2 = np.mean(np.abs(cdf_simulated_D2 - cdf_theoretical_D2))
    errors_D2.append(average_cdf_diff_D2)

    hist_D2_transformed, bins_D2_transformed = np.histogram(final_x_values_D2_transformed, bins=50, density=True)
    bin_centers_D2_transformed = (bins_D2_transformed[:-1] + bins_D2_transformed[1:]) / 2
    theoretical_density_D2_transformed = np.interp(bin_centers_D2_transformed, x_values, x_real)
    cdf_simulated_D2_transformed = np.cumsum(hist_D2_transformed) * (bins_D2_transformed[1] - bins_D2_transformed[0])
    cdf_theoretical_D2_transformed = np.cumsum(theoretical_density_D2_transformed) * (bins_D2_transformed[1] - bins_D2_transformed[0])
    cdf_simulated_D2_transformed /= cdf_simulated_D2_transformed[-1]
    cdf_theoretical_D2_transformed /= cdf_theoretical_D2_transformed[-1]
    average_cdf_diff_D2_transformed = np.mean(np.abs(cdf_simulated_D2_transformed - cdf_theoretical_D2_transformed))
    errors_D2_transformed.append(average_cdf_diff_D2_transformed)

    # D3 和 D3_transformed 的模拟
    
    

    
    final_x_values_D3 = np.array(final_x_values_D3).flatten()
    final_x_values_D3_transformed = np.array(final_x_values_D3_transformed).flatten()

    hist_D3, bins_D3 = np.histogram(final_x_values_D3, bins=50, density=True)
    bin_centers_D3 = (bins_D3[:-1] + bins_D3[1:]) / 2
    theoretical_density_D3 = np.interp(bin_centers_D3, x_values, x_real)
    cdf_simulated_D3 = np.cumsum(hist_D3) * (bins_D3[1] - bins_D3[0])
    cdf_theoretical_D3 = np.cumsum(theoretical_density_D3) * (bins_D3[1] - bins_D3[0])
    cdf_simulated_D3 /= cdf_simulated_D3[-1]
    cdf_theoretical_D3 /= cdf_theoretical_D3[-1]
    average_cdf_diff_D3 = np.mean(np.abs(cdf_simulated_D3 - cdf_theoretical_D3))
    errors_D3.append(average_cdf_diff_D3)

    hist_D3_transformed, bins_D3_transformed = np.histogram(final_x_values_D3_transformed, bins=50, density=True)
    bin_centers_D3_transformed = (bins_D3_transformed[:-1] + bins_D3_transformed[1:]) / 2
    theoretical_density_D3_transformed = np.interp(bin_centers_D3_transformed, x_values, x_real)
    cdf_simulated_D3_transformed = np.cumsum(hist_D3_transformed) * (bins_D3_transformed[1] - bins_D3_transformed[0])
    cdf_theoretical_D3_transformed = np.cumsum(theoretical_density_D3_transformed) * (bins_D3_transformed[1] - bins_D3_transformed[0])
    cdf_simulated_D3_transformed /= cdf_simulated_D3_transformed[-1]
    cdf_theoretical_D3_transformed /= cdf_theoretical_D3_transformed[-1]
    average_cdf_diff_D3_transformed = np.mean(np.abs(cdf_simulated_D3_transformed - cdf_theoretical_D3_transformed))
    errors_D3_transformed.append(average_cdf_diff_D3_transformed)
    end_time = time.time()
    print(f" delta_t = {fixed_dt:.4f}, Iteration Time = {end_time - start_time:.4f} seconds")

    




# 预定义颜色
colors = {
    'D1': '#1f77b4',  # 深蓝色
    'D1_transformed': '#aec7e8',  # 浅蓝色
    'D2': '#ff7f0e',  # 深橙色
    'D2_transformed': '#ffbb78',  # 浅橙色
    'D3': '#2ca02c',  # 深绿色
    'D3_transformed': '#98df8a',  # 浅绿色
}

# 绘制误差与 delta_t 的关系图
plt.figure()
plt.plot(fixed_dts, errors_D1, marker='o', color=colors['D1'], label='Error UBU')
plt.plot(fixed_dts, errors_D1_transformed, marker='o', color=colors['D1_transformed'], label='Error UBU_transformed')
plt.plot(fixed_dts, errors_D2, marker='o', color=colors['D2'], label='Error BAOAB')
plt.plot(fixed_dts, errors_D2_transformed, marker='o', color=colors['D2_transformed'], label='Error BAOAB_transformed')
plt.plot(fixed_dts, errors_D3, marker='o', color=colors['D3'], label='Error BUB')
plt.plot(fixed_dts, errors_D3_transformed, marker='o', color=colors['D3_transformed'], label='Error BUB_transformed')
plt.xlabel('Time Step (delta_t)')
plt.ylabel('Average CDF Difference (Error)')
plt.title('Error vs. Time Step (delta_t)')
plt.legend()
plt.show()
