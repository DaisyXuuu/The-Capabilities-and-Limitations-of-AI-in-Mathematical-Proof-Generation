#6.1.2参数估计

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 定义模型方程
def model(y, t, r, K, beta):
    C, A = y
    dC_dt = r * C * (1 - C / K) * A
    dA_dt = beta * C * (1 - A)
    return [dC_dt, dA_dt]

# 定义函数来计算C(t)和A(t)
def solve_odes(t, r, K, beta, C0, A0):
    y0 = [C0, A0]
    sol = odeint(model, y0, t, args=(r, K, beta))
    return sol[:, 0], sol[:, 1]

# 初始数据
time_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Time(Year)
C_data = np.array([10.00, 22.02, 40.28, 64.25, 93.40, 127.16, 164.78, 205.42, 248.00, 291.12, 333.33]) # AI capability
A_data = np.array([0.00, 0.0952, 0.1813, 0.2592, 0.3297, 0.3935, 0.4512, 0.5034, 0.5507, 0.5934, 0.6321]) #Adoption rate

# 定义误差函数来拟合参数
def error_function(params):
    r, K, beta = params
    C_sol, A_sol = solve_odes(time_data, r, K, beta, C_data[0], A_data[0])
    error_C = np.sum((C_sol - C_data) ** 2)
    error_A = np.sum((A_sol - A_data) ** 2)
    return error_C + 100000000*error_A

# 初始参数猜测
initial_guess = [0.15, 250, 0.05] #内在增长率 r： 这个值反映AI能力在理想情况下的增长速度。最大能力 K,采纳率常数 β


# 使用优化器进行参数拟合
from scipy.optimize import minimize
result = minimize(error_function, initial_guess, method='Nelder-Mead')

# 最优参数
r_estimated, K_estimated, beta_estimated = result.x
print(f"估计的参数：r = {r_estimated}, K = {K_estimated}, β = {beta_estimated}")

# 绘图比较模型与实际数据
C_fit, A_fit = solve_odes(time_data, r_estimated, K_estimated, beta_estimated, C_data[0], A_data[0])

# 分别绘制 C(t) 和 A(t)
plt.figure()

# 绘制 C(t)
plt.subplot(2, 1, 1)
plt.plot(time_data, C_data, 'o', label='True C(t)')
plt.plot(time_data, C_fit, '-', label='Fit C(t)')
plt.xlabel('Time (Year)')
plt.ylabel('AI capability C(t)')
plt.legend()


# 绘制 A(t)
plt.subplot(2, 1, 2)
plt.plot(time_data, A_data, 'o', label='True A(t)')
plt.plot(time_data, A_fit, '-', label='Fit A(t)')
plt.xlabel('Time (Year)')
plt.ylabel('Adoption rate A(t)')
plt.legend()

plt.tight_layout()
plt.show()

#%% 6.1.5 敏感度分析
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义模型方程
def model(y, t, r, K, beta):
    C, A = y
    dC_dt = r * C * (1 - C / K) * A
    dA_dt = beta * C * (1 - A)
    return [dC_dt, dA_dt]

# 定义求解函数
def solve_odes(t, r, K, beta, C0, A0):
    y0 = [C0, A0]
    sol = odeint(model, y0, t, args=(r, K, beta))
    return sol[:, 0], sol[:, 1]

# 初始参数和数据
time_data = np.linspace(0, 10, 100)
C0, A0 = 10, 0.1
r_fit, K_fit, beta_fit = 0.5657, 267.3989, 0.0096  # 最优参数估计

# 定义敏感性分析的扰动幅度
perturbation = 0.1  # 10% 的扰动

# 对 r 进行敏感性分析
r_values = [r_fit * (1 - perturbation), r_fit, r_fit * (1 + perturbation)]
C_r, A_r = [], []
for r in r_values:
    C, A = solve_odes(time_data, r, K_fit, beta_fit, C0, A0)
    C_r.append(C)
    A_r.append(A)

# 对 K 进行敏感性分析
K_values = [K_fit * (1 - perturbation), K_fit, K_fit * (1 + perturbation)]
C_K, A_K = [], []
for K in K_values:
    C, A = solve_odes(time_data, r_fit, K, beta_fit, C0, A0)
    C_K.append(C)
    A_K.append(A)

# 对 β 进行敏感性分析
beta_values = [beta_fit * (1 - perturbation), beta_fit, beta_fit * (1 + perturbation)]
C_beta, A_beta = [], []
for beta in beta_values:
    C, A = solve_odes(time_data, r_fit, K_fit, beta, C0, A0)
    C_beta.append(C)
    A_beta.append(A)

# 绘制敏感性分析结果
plt.figure(figsize=(12, 8))

# r 的敏感性分析图
plt.subplot(3, 2, 1)
for i, r in enumerate(r_values):
    plt.plot(time_data, C_r[i], label=f'r = {r:.3f}')
plt.xlabel('Time (Years)')
plt.ylabel('C(t) (AI Capability)')
plt.legend()
plt.title("Sensitivity Analysis on r for C(t)")

plt.subplot(3, 2, 2)
for i, r in enumerate(r_values):
    plt.plot(time_data, A_r[i], label=f'r = {r:.3f}')
plt.xlabel('Time (Years)')
plt.ylabel('A(t) (Adoption Rate)')
plt.legend()
plt.title("Sensitivity Analysis on r for A(t)")

# K 的敏感性分析图
plt.subplot(3, 2, 3)
for i, K in enumerate(K_values):
    plt.plot(time_data, C_K[i], label=f'K = {K:.1f}')
plt.xlabel('Time (Years)')
plt.ylabel('C(t) (AI Capability)')
plt.legend()
plt.title("Sensitivity Analysis on K for C(t)")

plt.subplot(3, 2, 4)
for i, K in enumerate(K_values):
    plt.plot(time_data, A_K[i], label=f'K = {K:.1f}')
plt.xlabel('Time (Years)')
plt.ylabel('A(t) (Adoption Rate)')
plt.legend()
plt.title("Sensitivity Analysis on K for A(t)")

# β 的敏感性分析图
plt.subplot(3, 2, 5)
for i, beta in enumerate(beta_values):
    plt.plot(time_data, C_beta[i], label=f'beta = {beta:.4f}')
plt.xlabel('Time (Years)')
plt.ylabel('C(t) (AI Capability)')
plt.legend()
plt.title("Sensitivity Analysis on β for C(t)")

plt.subplot(3, 2, 6)
for i, beta in enumerate(beta_values):
    plt.plot(time_data, A_beta[i], label=f'beta = {beta:.4f}')
plt.xlabel('Time (Years)')
plt.ylabel('A(t) (Adoption Rate)')
plt.legend()
plt.title("Sensitivity Analysis on β for A(t)")

plt.tight_layout()
plt.show()


