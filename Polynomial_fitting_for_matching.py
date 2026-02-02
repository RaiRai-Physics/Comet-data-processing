#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:42:35 2025

@author: bmr0043
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('Star_Spectrum_interpolated_extrapolated_wecfzst_232_with_ratio.csv')  # Replace with your file name
Wavelength = df['wavelength']
Ratio_1 =df['Ratio_1']
Ratio_2 =df['Ratio_2']

# --- Plot 1: Both Ratios in the same plot ---
plt.figure(figsize=(12, 10))
plt.plot(Wavelength, Ratio_1, label='Ratio_1')
plt.plot(Wavelength, Ratio_2, label='Ratio_2')
plt.xlabel('Wavelength')
plt.ylabel('Ratio Value')
plt.title('Wavelength vs Ratio_1 and Ratio_2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('Ratio_plot.png')

# --- Plot 2: Subplots for Ratio_1 and Ratio_2 ---
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

axs[0].plot(Wavelength, Ratio_1, color='blue')
axs[0].set_ylabel('Ratio_1')
axs[0].set_title('Wavelength vs Ratio_1')
axs[0].grid(True)
axs[0].set_xlim(3500,5000)

axs[1].plot(Wavelength, Ratio_2, color='green')
axs[1].set_xlabel('Wavelength')
axs[1].set_ylabel('Ratio_2')
axs[1].set_title('Wavelength vs Ratio_2')
axs[1].grid(True)
axs[1].set_xlim(3500,5000)

plt.tight_layout()
plt.show()
plt.savefig('Ratio_subplot.png')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from numpy.polynomial.polynomial import Polynomial

# Step 1: Input your data here

# Step 2: Choose degree of polynomial (start with 2 or 3)
degree = 3

# Step 3: Fit the polynomial
coeffs = Polynomial.fit(Wavelength, Ratio_1, degree).convert().coef
p = Polynomial(coeffs)

# Step 4: Predict using the model
predicted_values = p(Wavelength)

# Step 5: Accuracy metrics
r2 = r2_score(Ratio_1, predicted_values)
rmse = np.sqrt(mean_squared_error(Ratio_1, predicted_values))

print(f"Polynomial Coefficients: {coeffs}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Step 6: Plot the result
#wavelengths_fine = np.linspace(min(wavelengths), max(wavelengths), 500)
#fit_values = p(wavelengths_fine)

#plt.scatter(wavelengths, data_values, color='blue', label='Original Data')
#plt.plot(wavelengths_fine, fit_values, color='red', label=f'Polynomial Fit (deg {degree})')
plt.xlabel('Wavelength')
plt.ylabel('Data Value')
plt.title('Polynomial Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()
