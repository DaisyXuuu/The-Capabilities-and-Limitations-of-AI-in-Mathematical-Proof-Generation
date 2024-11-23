import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d

def model(y, t, r, K, beta):
    C, A = y
    dC_dt = r * C * (1 - C / K) * A
    dA_dt = beta * C * (1 - A)
    return [dC_dt, dA_dt]

r = 0.5000 
K = 205.7549
beta = 0.0010

C = 53  
A = 0.12
time_data = np.linspace(0, 10, 100) 

solution = odeint(model, [C, A], time_data, args=(r, K, beta))
C_pred, A_pred = solution[:, 0], solution[:, 1]

df = pd.DataFrame({
    'Time': time_data,
    'C(t)': C_pred,
    'A(t)': A_pred
})


print(df)

########################################################################
########################################################################

time_history = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
historical_C = 245 / (1 + np.exp(-0.5 * (time_history - 5)))  
historical_A = 0.9 / (1 + np.exp(-0.8 * (historical_C / 50 - 3))) 

interp_C = interp1d(time_history, historical_C, kind='linear', fill_value='extrapolate')
interp_A = interp1d(time_history, historical_A, kind='linear', fill_value='extrapolate')

historical_C_interp = interp_C(time_data)
historical_A_interp = interp_A(time_data)

rmse_C = np.sqrt(mean_squared_error(historical_C_interp, df['C(t)']))
r2_C = r2_score(historical_C_interp, df['C(t)'])

rmse_A = np.sqrt(mean_squared_error(historical_A_interp, df['A(t)']))
r2_A = r2_score(historical_A_interp, df['A(t)'])

print(f'RMSE for C(t): {rmse_C}, R² for C(t): {r2_C}')
print(f'RMSE for A(t): {rmse_A}, R² for A(t): {r2_A}')

#########################################################################
#########################################################################

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(df['Time'], df['C(t)'], label='Model C(t)', color='blue')
plt.plot(time_data, historical_C_interp, label='Historical C(t)', color='orange')
plt.title('AI Capability Over Time')
plt.xlabel('Time')
plt.ylabel('C(t)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['Time'], df['A(t)'], label='Model A(t)', color='green')
plt.plot(time_data, historical_A_interp, label='Historical A(t)', color='red')
plt.title('Adoption Rate Over Time')
plt.xlabel('Time')
plt.ylabel('A(t)')
plt.legend()

plt.tight_layout()
plt.show()



