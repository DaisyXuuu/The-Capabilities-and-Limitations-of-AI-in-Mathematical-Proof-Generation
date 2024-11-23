import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.5  # AI capability growth rate
K = 205  # Maximum carrying capacity
beta = 0.001  # Adoption rate constant

# Differential equations
def dC_dt(C, A):
    return r * C * (1 - C / K) * A

def dA_dt(C, A):
    return beta * C * (1 - A)

# Create a grid
C_vals = np.linspace(0, K, 20)
A_vals = np.linspace(0, 1.5, 20)
C, A = np.meshgrid(C_vals, A_vals)

# Compute direction field
dC = dC_dt(C, A)
dA = dA_dt(C, A)

# Normalize the direction vectors
magnitude = np.sqrt(dC**2 + dA**2)
dC /= magnitude
dA /= magnitude

# Plot the direction field
plt.figure(figsize=(10, 6))
plt.quiver(C, A, dC, dA, color='black', angles='xy', scale=30)

# Mark equilibrium points with larger solid red dots
plt.scatter([0, 0, K], [0, 1, 1], color='red', s=200, label='Equilibrium points')

# Annotate equilibrium points
plt.text(0, 0, '(0, 0)', fontsize=12, color='red', ha='left', va='bottom')
plt.text(0, 1, '(0, 1)', fontsize=12, color='red', ha='left', va='bottom')
plt.text(K, 1, f'({int(K)}, 1)', fontsize=12, color='red', ha='right', va='bottom')

# Add titles and labels
plt.title("Direction Field of the System", fontsize=14)
plt.xlabel("C (AI Capability)", fontsize=12)
plt.ylabel("A (Adoption Rate)", fontsize=12)
plt.xlim(0, K)
plt.ylim(0, 1.5)
plt.legend(fontsize=12)
plt.grid()

# Display the plot
plt.show()

#%%
# 定义 F(C, A) 的守恒量函数
def F(C, A, beta=0.001, r=0.5, K=205):
    return (beta / r) * (np.log(A) - A) - C + (C**2) / (2 * K)

# 创建 C 和 A 的网格
C_vals = np.linspace(0.1, 10, 100)  # 避免 log(0) 问题
A_vals = np.linspace(0.1, 1, 100)
C, A = np.meshgrid(C_vals, A_vals)

# 计算 F(C, A)
F_vals = F(C, A)

# 绘制等值线图
plt.figure(figsize=(10, 6))
contour = plt.contour(C, A, F_vals, levels=20, cmap="viridis")
plt.clabel(contour, inline=True, fontsize=8)

# 标注平衡点
plt.scatter([0, 0, 10], [0, 1, 1], color='red', s=100, label='Equilibrium points')
plt.text(0, 0, '(0, 0)', fontsize=12, color='red', ha='left', va='bottom')
plt.text(0, 1, '(0, 1)', fontsize=12, color='red', ha='left', va='bottom')
plt.text(10, 1, '(10, 1)', fontsize=12, color='red', ha='right', va='bottom')

# 图形设置
plt.title("Contour Plot of F(C, A) = Constant", fontsize=14)
plt.xlabel("C (AI Capability)", fontsize=12)
plt.ylabel("A (Adoption Rate)", fontsize=12)
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.grid()

# 显示图形
plt.show()
