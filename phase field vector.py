import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of ODEs
def model(t, y, r, K, beta, c_a, c_b):
    C, A = y  # Unpack AI capability (C) and public acceptance (A)

    # AI capability dynamics with acceptance multiplier
    dC_dt = r * C * (1 - C / K) * A

    # Public acceptance dynamics
    dA_dt = beta * (C - c_a) * (C - c_b) * (1 - A)

    return [dC_dt, dA_dt]

# Parameters (fitted values)
r_fit = 0.5000    # Growth rate of AI capability
K_fit = 210  # Maximum AI capability (technical limit)
beta_fit = 0.00010 # Scaling factor for acceptance dynamics
c_a = 50.0        # First threshold for acceptance dynamics
c_b = 150.0       # Second threshold for acceptance dynamics

# Initial conditions
C0 = 5.0   # Initial AI capability (low starting point)
A0 = 0.2   # Initial public acceptance (low starting point)

# Time span for simulation
t_span = (0, 20)  # Simulate from t=0 to t=500 to capture long-term behavior
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points for evaluation


# Solve the system of ODEs
solution = solve_ivp(
    model,
    t_span,
    [C0, A0],
    args=(r_fit, K_fit, beta_fit, c_a, c_b),
    t_eval=t_eval,
    method='RK45'
)

# Extract the solution
C = solution.y[0]  # AI capability over time
A = solution.y[1]  # Public acceptance over time
t = solution.t     # Time points

# Plot the results
plt.figure(figsize=(12, 8))

# Plot AI capability over time
plt.subplot(2, 1, 1)
plt.plot(t, C, label='AI Capability (C)', color='blue')
plt.axhline(K_fit, color='green', linestyle='--', label='Capability Limit (K)')
plt.title('AI Capability and Public Acceptance Dynamics')
plt.xlabel('Time')
plt.ylabel('AI Capability (C)')
plt.legend()

# Plot public acceptance over time
plt.subplot(2, 1, 2)
plt.plot(t, A, label='Public Acceptance (A)', color='orange')
plt.xlabel('Time')
plt.ylabel('Mathematicians adoptation (A)')
plt.legend()

plt.tight_layout()
plt.show()
