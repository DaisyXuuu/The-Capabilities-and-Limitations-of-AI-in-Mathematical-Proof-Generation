import numpy as np
import matplotlib.pyplot as plt

# Parameters
r_fit = 0.5
K_fit = 210
beta_fit = 0.0001
c_a = 50.0
c_b = 150.0

# Define the vector field
def dC_dA(C, A, r, K, beta, c_a, c_b):
    dC = r * C * (1 - C / K) * A
    dA = beta * (C - c_a) * (C - c_b) * (1 - A)
    return dC, dA

# Grid for the vector field
C = np.linspace(0, 250, 20)
A = np.linspace(-1.5, 1.5, 20)
C_grid, A_grid = np.meshgrid(C, A)

# Compute the vector field
dC, dA = dC_dA(C_grid, A_grid, r_fit, K_fit, beta_fit, c_a, c_b)

# Normalize the vectors for better visualization
magnitude = np.sqrt(dC**2 + dA**2)
dC_norm = dC / magnitude
dA_norm = dA / magnitude

# Plot the vector field
plt.figure(figsize=(10, 6))
plt.quiver(C_grid, A_grid, dC_norm, dA_norm, magnitude, cmap='viridis', scale=30)
plt.title('Phase Space Vector Field of AI Capability and Mathematicians Adoption')
plt.xlabel('AI Capability (C)')
plt.ylabel('Mathematicians Adoption (A)')
plt.colorbar(label='Vector Magnitude')
plt.grid()
plt.show()
