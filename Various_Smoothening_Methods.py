#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:06:47 2025

@author: bmr0043
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# --- Load data ---
df = pd.read_csv('Star_Spectrum_interpolated_extrapolated_wecfzst_232_with_ratio.csv')
wavelength_all = df['wavelength']
raw_sens_counts_per_sec_all = df['Ratio_2']

# Trim unwanted start values (as before)
wavelength = wavelength_all[302:]
raw_sens_counts = raw_sens_counts_per_sec_all[302:]

# --- Savitzky-Golay ---
savgol_smoothed = savgol_filter(raw_sens_counts, window_length=101, polyorder=5)

# --- Median filter ---
median_smoothed = median_filter(raw_sens_counts, size=101)

# --- Polynomial fit (5th degree) ---
poly_coeffs = np.polyfit(wavelength, raw_sens_counts, deg=5)
poly_smoothed = np.polyval(poly_coeffs, wavelength)

# --- Univariate spline ---
spline = UnivariateSpline(wavelength, raw_sens_counts, s=50)  # try s=1–100
spline_smoothed = spline(wavelength)

# --- Asymmetric Least Squares (ALS) ---
def asymmetric_least_squares(y, lam=1e6, p=0.001, niter=100):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2)).toarray()
    D = sparse.csc_matrix(D)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

als_smoothed = asymmetric_least_squares(raw_sens_counts.values, lam=1e6, p=0.005)

# --- Plot everything ---
plt.figure(figsize=(12, 7))

plt.plot(wavelength, raw_sens_counts, label='Raw', color='gray', alpha=0.5)
plt.plot(wavelength, savgol_smoothed, label='Savitzky-Golay', color='blue')
plt.plot(wavelength, median_smoothed, label='Median Filter', color='green')
plt.plot(wavelength, poly_smoothed, label='Polynomial Fit', color='red', linestyle='--')
plt.plot(wavelength, spline_smoothed, label='Spline Fit', color='orange')
plt.plot(wavelength, als_smoothed, label='ALS (Best for continuum)', color='purple', linestyle='-.')

plt.title('Comparison of Smoothing / Continuum Fitting Techniques')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Sensitivity Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Comparison_of_different_Smoothening_techniques_counts_per_sec.png', dpi=300)
plt.show()

#Saving things to a csv file
df_1 = pd.DataFrame({
    'Wavelength':wavelength,
    'Raw Counts':raw_sens_counts,
    'Savitzsky-Golay':savgol_smoothed,
    'Median Filter':median_smoothed,
    'Polynomial Fit':poly_smoothed,
    'Spline Fit':spline_smoothed,
    'ALS Fit':als_smoothed
        })
df_1.to_csv('Smoothed_Flux_to_counts_per_sec_wecfzst_232.csv', index=False)

#Seeing which is a better fit
methods = {
    'Savitzky-Golay': savgol_smoothed,
    'Median Filter': median_smoothed,
    'Polynomial Fit': poly_smoothed,
    'Spline Fit': spline_smoothed,
    'ALS Fit': als_smoothed
}

for name, fit in methods.items():
    mse = mean_squared_error(raw_sens_counts, fit)
    r2 = r2_score(raw_sens_counts, fit)
    corr, _ = pearsonr(raw_sens_counts, fit)
    print(f"{name:20s} | MSE: {mse:.5e} | R²: {r2:.5f} | Corr: {corr:.5f}")

#Finding the min and max values in the difference between fitting and exact data
df_2 = pd.read_csv('Smoothed_Flux_to_counts_per_sec_wecfzst_232_with_diff_ratio_2.csv')
Poly_diff =df_2['Diff_Polynomial'].abs()
Sav_Gol_diff =df_2['Diff_Sav-Gol'].abs()
Median_diff=df_2['Diff_Median'].abs()
Spline_diff=df_2['Diff_Spline'].abs()
ALS_diff=df_2['Diff_ALS'].abs()

# Get min and max of the absolute values
min_val_poly = Poly_diff.min()
max_val_poly = Poly_diff.max()
print("Min absolute value Polynomial diff:", min_val_poly)
print("Max absolute value Polynomial diff:", max_val_poly)

min_val_Sav_Gol = Sav_Gol_diff.min()
max_val_Sav_Gol = Sav_Gol_diff.max()
print("Min absolute value Sav_Gol diff:", min_val_Sav_Gol)
print("Max absolute value Sav_Gol diff:", max_val_Sav_Gol)

min_val_Median = Median_diff.min()
max_val_Median = Median_diff.max()
print("Min absolute value Median diff:", min_val_Median)
print("Max absolute value Median diff:", max_val_Median)

min_val_Spline = Spline_diff.min()
max_val_Spline = Spline_diff.max()
print("Min absolute value Spline diff:", min_val_Spline)
print("Max absolute value Spline diff:", max_val_Spline)

min_val_ALS_diff = ALS_diff.min()
max_val_ALS_diff = ALS_diff.max()
print("Min absolute value ALS diff:", min_val_ALS_diff)
print("Max absolute value ALS diff:", max_val_ALS_diff)


