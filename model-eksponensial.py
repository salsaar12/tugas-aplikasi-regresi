import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Create the dataset
data = pd.DataFrame({
    'Hours Studied': [8, 2, 5, 1, 9],
    'Previous Scores': [52, 53, 93, 99, 83],
    'Extracurricular Activities': ['No', 'No', 'Yes', 'No', 'Yes'],
    'Sleep Hours': [7, 5, 6, 7, 6],
    'Sample Question Papers Practiced': [1, 2, 5, 3, 9],
    'Performance Index': [48.0, 24.0, 80.0, 74.0, 85.0]
})

# Extract relevant columns
TB = data['Hours Studied'].values.reshape(-1, 1)
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
NT = data['Performance Index'].values

# Exponential Model
def exponential_model(x, a, b):
    return a * np.exp(b * x)

params_tb_exp, _ = curve_fit(exponential_model, TB.flatten(), NT)
params_nl_exp, _ = curve_fit(exponential_model, NL.flatten(), NT)

# Predictions
NT_pred_exp_tb = exponential_model(TB, *params_tb_exp)
NT_pred_exp_nl = exponential_model(NL, *params_nl_exp)

# Plotting the results
plt.figure(figsize=(14, 7))

# Problem 1: Hours Studied vs Performance Index
plt.subplot(1, 2, 1)
plt.scatter(TB, NT, label='Data', color='blue')
plt.plot(TB, NT_pred_exp_tb, label='Exponential Model', color='orange')
plt.xlabel('Hours Studied (TB)')
plt.ylabel('Performance Index (NT)')
plt.title('Hours Studied vs Performance Index')
plt.legend()

# Problem 2: Sample Question Papers Practiced vs Performance Index
plt.subplot(1, 2, 2)
plt.scatter(NL, NT, label='Data', color='blue')
plt.plot(NL, NT_pred_exp_nl, label='Exponential Model', color='orange')
plt.xlabel('Sample Question Papers Practiced (NL)')
plt.ylabel('Performance Index (NT)')
plt.title('Sample Question Papers Practiced vs Performance Index')
plt.legend()

plt.tight_layout()
plt.show()