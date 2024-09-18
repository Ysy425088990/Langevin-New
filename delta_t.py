import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad
import time



def matrix_exponential(matrix, t):
    return expm(matrix * t)




# 使用 D1 进行模拟
def run_simulation_D1(delta_t, end_t, gamma, beta, second_iteration, D1, initial):
    D = D1
    h = delta_t / 2
    n_t = int(end_t / delta_t)
    
    # 初始化变量
    x = np.zeros(n_t + 1)
    p = np.zeros(n_t + 1)

    # 使用给定的 initial 初始化 x[0]
    x[0],p[0] = initial
    
    
    for i in range(n_t):
        # 第一次迭代使用矩阵指数公式
        W_h = np.random.normal(0, np.sqrt(h))
        exp_Dh = matrix_exponential(D, h)
        
        update_vector = exp_Dh @ np.array([x[i], p[i]]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
        x_mid, p_mid = update_vector
        
        # 第二次迭代使用前向欧拉公式
        x_mid, p_mid = second_iteration(x_mid, p_mid, delta_t, beta)
        
        # 第三次迭代使用矩阵指数公式
        W_h = np.random.normal(0, np.sqrt(h))
        update_vector = exp_Dh @ np.array([x_mid, p_mid]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
        if i + 1 < len(x):  # 确保不会超出范围
            x[i + 1], p[i + 1] = update_vector
        else:
            break
    
    return np.array(x)

# 使用 D2 进行模拟
def run_simulation_D2(delta_t, end_t, gamma, beta, second_iteration, D2, initial):
    D = D2
    h = delta_t / 2
    n_t = int(end_t / delta_t)
    
    # 初始化变量
    x = np.zeros(n_t + 1)
    p = np.zeros(n_t + 1)

    # 使用给定的 initial 初始化 x[0]
    x[0],p[0] = initial
    
    
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
    
    return np.array(x)

# 使用 D3 进行模拟
def run_simulation_D3(delta_t, end_t, gamma, beta, second_iteration, D3, initial):
    D = D3
    h = delta_t / 2
    n_t = int(end_t / delta_t)
    
    # 初始化变量
    x = np.zeros(n_t + 1)
    p = np.zeros(n_t + 1)

    # 使用给定的 initial 初始化 x[0]
    x[0],p[0] = initial
    
    
    for i in range(n_t):
        # 第一次迭代使用前向欧拉公式（原来的第二次迭代）
        x_mid, p_mid = second_iteration(x[i], p[i], h, beta)
        
        # 第二次迭代使用矩阵指数公式（原来的第一次迭代）
        W_h = np.random.normal(0, np.sqrt(delta_t))
        exp_Dh = matrix_exponential(D, delta_t)
        update_vector = exp_Dh @ np.array([x_mid, p_mid]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
        x_mid, p_mid = update_vector
        
        # 第三次迭代使用前向欧拉公式（原来的第二次迭代）
        x_mid, p_mid = second_iteration(x_mid, p_mid, h, beta)
        
        if i + 1 < len(x):  # 确保不会超出范围
            x[i + 1], p[i + 1] = x_mid, p_mid
        else:
            break
    
    return np.array(x)
