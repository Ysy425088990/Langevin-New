import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad
import time



def matrix_exponential(matrix, t):
    return expm(matrix * t)



def run_simulation_D1_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D1, initial,k_max,k_function):
    D = D1
    
    # 初始化变量
    x = []
    p = []
    clock = []
    # 使用给定的 initial 初始化 x[0]
    x_k,p_k = initial
    
    tau = 0
    
    for i in range(k_max+1):
        delta_t = fixed_dt
        h = delta_t / 2
        W_h = np.random.normal(0, np.sqrt(h))
        exp_Dh = matrix_exponential(D, h)
        
        update_vector = exp_Dh @ np.array([x_k, p_k]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
        x_mid, p_mid = update_vector
        
        x_mid, p_mid = second_iteration_transformed(x_mid, p_mid, delta_t, beta)
        
        W_h = np.random.normal(0, np.sqrt(h))
        update_vector = exp_Dh @ np.array([x_mid, p_mid]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
        x_k,p_k = update_vector
        tau = tau + delta_t/k_function(update_vector[0])
        
        if tau > end_t :  # 确保不会超出范围
            x.append(update_vector[0])
            p.append(update_vector[1])
            clock.append(tau)
            
        
    
    
    
    return np.array(x),np.array(clock)


def run_simulation_D2_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D2, initial,k_max,k_function):
    D = D2
    
    # 初始化变量
    x = []
    p = []
    clock = []
    # 使用给定的 initial 初始化 x[0]
    x_k,p_k = initial
    
    tau = 0
    
    for i in range(k_max+1):
        delta_t = fixed_dt
        h = delta_t / 2
        x_mid, p_mid = second_iteration_transformed(x_k, p_k, h, beta)
        
        x_mid += p_mid * h
        p_mid += 0
        
        W_h = np.random.normal(0, np.sqrt(delta_t))
        exp_Dh = matrix_exponential(D, delta_t)
        update_vector = exp_Dh @ np.array([x_mid, p_mid]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
        x_mid, p_mid = update_vector
        
        x_mid += p_mid * h
        p_mid += 0
        
        x_mid, p_mid = second_iteration_transformed(x_mid, p_mid, h, beta)
        x_k,p_k = x_mid,p_mid
        tau = tau + delta_t/k_function(update_vector[0])
        
        if tau > end_t :  # 确保不会超出范围
            x.append(x_k)
            p.append(p_k)
            clock.append(tau)
            
        
    
    
    
    return np.array(x),np.array(clock)
    
    
    

def run_simulation_D3_transformed(fixed_dt, end_t, gamma, beta, second_iteration_transformed, D3, initial,k_max,k_function):
    D = D3
    
    # 初始化变量
    x = []
    p = []
    clock = []
    # 使用给定的 initial 初始化 x[0]
    x_k,p_k = initial
    
    tau = 0
    
    for i in range(k_max+1):
        delta_t = fixed_dt
        h = delta_t / 2
        x_mid, p_mid = second_iteration_transformed(x_k, p_k, h, beta)
        
        W_h = np.random.normal(0, np.sqrt(delta_t))
        exp_Dh = matrix_exponential(D, delta_t)
        update_vector = exp_Dh @ np.array([x_mid, p_mid]) + exp_Dh @ np.array([0, np.sqrt(2 * gamma / beta)]) * W_h
        x_mid, p_mid = update_vector
        
        x_mid, p_mid = second_iteration_transformed(x_mid, p_mid, h, beta)
        x_k,p_k = x_mid,p_mid
        tau = tau + delta_t/k_function(update_vector[0])
        
        if tau > end_t :  # 确保不会超出范围
            x.append(x_k)
            p.append(p_k)
            clock.append(tau)
            
        
    
    
    
    return np.array(x),np.array(clock)
