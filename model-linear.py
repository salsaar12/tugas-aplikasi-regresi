import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

# Linear Model
linear_model_tb = LinearRegression().fit(TB, NT)
linear_model_nl = LinearRegression().fit(NL, NT)

# Predictions
NT_pred_linear_tb = linear_model_tb.predict(TB)
NT_pred_linear_nl = linear_model_nl.predict(NL)

# Plotting the results
plt.figure(figsize=(14, 7))

# Problem 1: Hours Studied vs Performance Index
plt.subplot(1, 2, 1)
plt.scatter(TB, NT, label='Data', color='blue')
plt.plot(TB, NT_pred_linear_tb, label='Linear Model', color='red')
plt.xlabel('Hours Studied (TB)')
plt.ylabel('Performance Index (NT)')
plt.title('Hours Studied vs Performance Index')
plt.legend()

# Problem 2: Sample Question Papers Practiced vs Performance Index
plt.subplot(1, 2, 2)
plt.scatter(NL, NT, label='Data', color='blue')
plt.plot(NL, NT_pred_linear_nl, label='Linear Model', color='red')
plt.xlabel('Sample Question Papers Practiced (NL)')
plt.ylabel('Performance Index (NT)')
plt.title('Sample Question Papers Practiced vs Performance Index')
plt.legend()

plt.tight_layout()
plt.show()