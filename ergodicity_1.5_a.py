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
num_simulations = 1  # 运行的模拟次数
x_start = -10  # x 轴起始位置
x_end = 10     # x 轴结束位置
num_x_steps = 4000  # x 轴的步数
end_t = 0
T = 250
fixed_dt = 0.1  # 时间步长
k_max = np.int(T/fixed_dt)
# 动态时间步长函数
def dynamic_time_step(x_prev, fixed_dt):
    return fixed_dt * k_function(x_prev)  # 可以根据需要调整

# 定义矩阵 D1 和 D2
D1 = np.array([[0, 1], [0, -gamma]])
D2 = np.array([[0, 0], [0, -gamma]])
D3 = np.array([[0, 1], [0, -gamma]])

# 定义势函数和函数 k(x)
def potential(x):
    return 0.5 * np.sqrt(np.abs(x) ** 3)

def k_function(x):
    return np.sqrt(np.abs(x)) # k(x) 为常数

# 计算边缘分布的归一化常数
def normalizing_constant():
    integrand = lambda x: np.exp(-beta * potential(x))
    integral, _ = quad(integrand, -np.inf, np.inf)
    return 1 / integral

# 计算边缘分布
def rho_eq(x):
    Z = normalizing_constant()
    return Z * np.exp(-beta * potential(x))

# 手动计算势函数的梯度
def grad_potential(x):
    return 1.5 * 0.5 * np.sign(x) * np.sqrt(np.abs(x))

# 手动计算 k(x) 的对数梯度
def grad_log_k_function(x):
    return 0.5 * 1/np.sqrt(np.abs(x)) * np.sign(x)

# 第二次迭代
def second_iteration_transformed(x, p, delta_t, beta):
    grad_potential_x = grad_potential(x)
    grad_log_k_x = grad_log_k_function(x)
    dp = -(grad_potential_x - (1 / beta) * grad_log_k_x) * delta_t
    p_new = p + dp
    x_new = x
    return x_new, p_new

def second_iteration(x, p, delta_t, beta):
    grad_potential_x = grad_potential(x)
    dp = -(grad_potential_x) * delta_t
    p_new = p + dp
    x_new = x
    return x_new, p_new

# 计算 x_real 分布
x_values = np.linspace(x_start, x_end, num_x_steps)
x_real = np.array([rho_eq(x) for x in x_values])

# 运行模拟并保留所有迭代数据
initial = [1000,0]

x_values_D1 = run_simulation_D1(fixed_dt, T, gamma, beta, second_iteration, D1, initial)
x_values_D2 = run_simulation_D2(fixed_dt, T, gamma, beta, second_iteration, D2, initial)
x_values_D3 = run_simulation_D3(fixed_dt, T, gamma, beta, second_iteration, D3, initial)
x_values_D1_transformed = run_simulation_D1_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D1,  initial,k_max,k_function)[0]
x_values_D2_transformed = run_simulation_D2_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D2,  initial,k_max,k_function)[0]
x_values_D3_transformed = run_simulation_D3_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D3,  initial,k_max,k_function)[0]
# 绘制图形
plt.figure(figsize=(12, 8))
plt.plot(x_values_D1, label='x values D1')
plt.plot(x_values_D2, label='x values D2')
plt.plot(x_values_D3, label='x values D3')
plt.plot(x_values_D1_transformed, label='x values D1 Transformed', linestyle='--')
plt.plot(x_values_D2_transformed, label='x values D2 Transformed', linestyle='--')
plt.plot(x_values_D3_transformed, label='x values D3 Transformed', linestyle='--')

plt.xlabel('Iteration')
plt.ylabel('x values')
plt.title('x values over iterations')
plt.legend()
plt.show()