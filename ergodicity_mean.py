import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
from delta_t import run_simulation_D1, run_simulation_D2, run_simulation_D3
from adapt_deltat import run_simulation_D1_transformed, run_simulation_D2_transformed, run_simulation_D3_transformed



# 定义参数
gamma = 1
beta = 1
x_start = -10
x_end = 10
num_x_steps = 4000
T = 150
end_t = 0
fixed_dt = 0.1
k_max = np.int(T/fixed_dt)
iteration_start = 1000
iteration_end = 1500
a = 1000000  # 用来定义初始值 x_0^2 + p_0^2 = a

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

# 定义多组初始值 (x_0, p_0)
num_initial_conditions = 1000  # 初始值的组数
initial_conditions = []
for _ in range(num_initial_conditions):
    # 随机生成满足 x_0^2 + p_0^2 = a 的初始值
    theta = np.random.uniform(0, 2 * np.pi)
    x_0 = np.sqrt(a) * np.cos(theta)
    p_0 = np.sqrt(a) * np.sin(theta)
    initial_conditions.append((x_0, p_0))
    


# 模拟并计算每组初始值的均值
mean_values = []
for x_0, p_0 in initial_conditions: 
    initial = [x_0,p_0]
    x_values_D1 = run_simulation_D1_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D1,  initial,k_max,k_function)[0]
    x_values_D2 = run_simulation_D2_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D2,  initial,k_max,k_function)[0]
    x_values_D3 = run_simulation_D3_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D3,  initial,k_max,k_function)[0]

    # 计算 iteration 1000 到 1500 之间的均值
    mean_D1 = np.mean(x_values_D1[iteration_start:iteration_end])
    mean_D2 = np.mean(x_values_D2[iteration_start:iteration_end])
    mean_D3 = np.mean(x_values_D3[iteration_start:iteration_end])
    
    # 将均值和初始值保存
    mean_values.append((x_0, p_0, mean_D1, mean_D2, mean_D3))

# 将均值展示在三维图上
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x_0_vals = [val[0] for val in mean_values]
p_0_vals = [val[1] for val in mean_values]
mean_D1_vals = [val[2] for val in mean_values]
mean_D2_vals = [val[3] for val in mean_values]
mean_D3_vals = [val[4] for val in mean_values]

# 绘制每个算法的均值曲面
ax.scatter(x_0_vals, p_0_vals, mean_D1_vals, label='Mean D1', color='r')
ax.scatter(x_0_vals, p_0_vals, mean_D2_vals, label='Mean D2', color='g')
ax.scatter(x_0_vals, p_0_vals, mean_D3_vals, label='Mean D3', color='b')

ax.set_xlabel('Initial x_0')
ax.set_ylabel('Initial p_0')
ax.set_zlabel('Mean Value')
plt.title('Mean Values of x in Iteration Range [1000, 1500]')
plt.legend()
plt.show()
