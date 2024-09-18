import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad
import time
from delta_t import run_simulation_D1, run_simulation_D2, run_simulation_D3
from adapt_deltat import run_simulation_D1_transformed, run_simulation_D2_transformed, run_simulation_D3_transformed

# 定义参数
gamma = 1
beta = 1  # 假设 beta 为常数
num_delta_t = 6
num_simulations = 100  # 运行的模拟次数
x_start = -10  # x 轴起始位置
x_end = 10     # x 轴结束位置
num_x_steps = 400  # x 轴的步数
end_t = 100
sample_number = 1

# 定义不同的时间步长
fixed_dts = np.zeros(num_delta_t)
for i in range(num_delta_t):
    fixed_dts[i] = 0.05 * 2 ** (i)
    
# 动态时间步长函数
def dynamic_time_step(x_prev, fixed_dt):
    return fixed_dt * k_function(x_prev) # 可以根据需要调整

# 定义矩阵 D1 和 D2
D1 = np.array([[0, 1], [0, -gamma]])
D2 = np.array([[0, 0], [0, -gamma]])
D3 = np.array([[0, 1], [0, -gamma]])

def matrix_exponential(matrix, t):
    return expm(matrix * t)

# 定义势函数和函数 k(x)
def potential(x):
    return np.sqrt(np.abs(x**3))

def k_function(x):
    return np.exp(- 0.1 * np.sqrt(np.abs(x**3)))

def normalizing_constant():
    integrand = lambda x: np.exp(-beta * potential(x))
    integral, _ = quad(integrand, -np.inf, np.inf)
    return 1 / integral

def rho_eq(x):
    Z = normalizing_constant()
    return Z * np.exp(-beta * potential(x))

def compute_x_real(x_values):
    return np.array([rho_eq(x) for x in x_values])

# 手动计算势函数的梯度
def grad_potential(x):
    return np.sign(x) * 1.5 * np.sqrt(np.abs(x))

# 手动计算 k(x) 的对数梯度
def grad_log_k_function(x):
    return - np.sign(x) * 0.15 * np.sqrt(np.abs(x))

# 计算边缘分布
def rho_eq(x):
    Z = normalizing_constant()
    return Z * np.exp(-beta * potential(x))

# 计算理论 PDF
def compute_theoretical_pdf(x_values):
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
x_real = compute_theoretical_pdf(x_values)

# 初始化错误列表
errors_D1 = []
errors_D2 = []
errors_D3 = []
errors_D1_transformed = []
errors_D2_transformed = []
errors_D3_transformed = []

for fixed_dt in fixed_dts:
    start_time = time.time()
    
    # D1 和 D1_transformed 的模拟
    final_x_values_D1 = []
    final_x_values_D1_transformed = []
    final_x_values_D2 = []
    final_x_values_D2_transformed = []
    final_x_values_D3 = []
    final_x_values_D3_transformed = []

    for _ in range(num_simulations):
        initial = np.random.normal(0, 1)
        x_D1 = run_simulation_D1(fixed_dt, end_t, gamma, beta, second_iteration, D1, initial)
        final_x_values_D1.append(x_D1[-sample_number:])  # 只保留最后 20 个数据点
        
        x_D1_transformed = run_simulation_D1_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D1, dynamic_time_step, initial)
        final_x_values_D1_transformed.append(x_D1_transformed[-sample_number:])
        
        x_D2 = run_simulation_D2(fixed_dt, end_t, gamma, beta, second_iteration, D2, initial)
        final_x_values_D2.append(x_D2[-sample_number:])
        
        x_D2_transformed = run_simulation_D2_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D2, dynamic_time_step, initial)
        final_x_values_D2_transformed.append(x_D2_transformed[-sample_number:])
        
        x_D3 = run_simulation_D3(fixed_dt, end_t, gamma, beta, second_iteration, D3, initial)
        final_x_values_D3.append(x_D3[-sample_number:])
        
        x_D3_transformed = run_simulation_D3_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D3, dynamic_time_step, initial)
        final_x_values_D3_transformed.append(x_D3_transformed[-sample_number:])
    
    final_x_values_D1 = np.array(final_x_values_D1).flatten()
    final_x_values_D1_transformed = np.array(final_x_values_D1_transformed).flatten()
    final_x_values_D2 = np.array(final_x_values_D2).flatten()
    final_x_values_D2_transformed = np.array(final_x_values_D2_transformed).flatten()
    final_x_values_D3 = np.array(final_x_values_D3).flatten()
    final_x_values_D3_transformed = np.array(final_x_values_D3_transformed).flatten()

    def compute_pdf_and_error(final_x_values, x_values, theoretical_density):
        # 计算模拟 PDF
        hist, bins = np.histogram(final_x_values, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 计算模拟 PDF 并确保 bin_centers 和 hist 一致
        simulated_pdf = np.interp(x_values, bin_centers, hist, left=0, right=0)
        
        # 计算理论 PDF
        theoretical_pdf = np.interp(x_values, x_values, theoretical_density)
        
        # 计算误差
        error = np.mean(np.abs(simulated_pdf - theoretical_pdf))
        return error

    # 计算误差
    errors_D1.append(compute_pdf_and_error(final_x_values_D1, x_values, x_real))
    errors_D1_transformed.append(compute_pdf_and_error(final_x_values_D1_transformed, x_values, x_real))
    errors_D2.append(compute_pdf_and_error(final_x_values_D2, x_values, x_real))
    errors_D2_transformed.append(compute_pdf_and_error(final_x_values_D2_transformed, x_values, x_real))
    errors_D3.append(compute_pdf_and_error(final_x_values_D3, x_values, x_real))
    errors_D3_transformed.append(compute_pdf_and_error(final_x_values_D3_transformed, x_values, x_real))

    end_time = time.time()
    print(f"delta_t = {fixed_dt:.4f}, Iteration Time = {end_time - start_time:.4f} seconds")

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
plt.plot(fixed_dts, errors_D1, marker='o', color=colors['D1'], label='Error D1')
plt.plot(fixed_dts, errors_D1_transformed, marker='o', color=colors['D1_transformed'], label='Error D1_transformed')
plt.plot(fixed_dts, errors_D2, marker='o', color=colors['D2'], label='Error D2')
plt.plot(fixed_dts, errors_D2_transformed, marker='o', color=colors['D2_transformed'], label='Error D2_transformed')
plt.plot(fixed_dts, errors_D3, marker='o', color=colors['D3'], label='Error D3')
plt.plot(fixed_dts, errors_D3_transformed, marker='o', color=colors['D3_transformed'], label='Error D3_transformed')

# 设置对数刻度
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Time Step (delta_t)')
plt.ylabel('Average PDF Difference (Error)')
plt.title('Error vs. Time Step (delta_t)')
plt.legend()
plt.show()
