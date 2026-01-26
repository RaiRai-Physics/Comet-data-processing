#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 20:35:09 2025

@author: bmr0043
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:14:48 2025

@author: bmr0043
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:07:19 2025

@author: bmr0043
"""
"This is for the wcfsto file 0232"

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# STEP 1: Read your FITS file 
hdul1 = fits.open('wecfsto_0232_178822nd_23-12-2024.fits')
data1 = hdul1[0].data
header1 = hdul1[0].header

# STEP 2: Reconstruct wavelength array from FITS header
crval1 = header1['CRVAL1']  # starting wavelength
cdelt1 = header1['CDELT1']  # step size
npix = len(data1)
wavelength_star = crval1 + cdelt1 * np.arange(npix)

# STEP 3: Create star flux array (replace with real flux values!)
df = pd.read_csv("formatted_flux_data.csv")  
wavelength = df["Wavelength_A"].values
flux = df["Flux_erg_cm2_s1_A1"].values

real_star_wavelength = wavelength
real_star_flux = flux

# STEP 4: Interpolate star flux to match comet wavelength scale
# You can use 'linear', 'cubic', or 'quadratic'
interp_func_linear = interp1d(real_star_wavelength, real_star_flux, kind='linear', bounds_error=False, fill_value="extrapolate")
interpolated_star_flux_linear = interp_func_linear(wavelength_star)
interp_func_quadratic= interp1d(real_star_wavelength, real_star_flux, kind='quadratic', bounds_error=False, fill_value="extrapolate")
interpolated_star_flux_quadratic = interp_func_quadratic(wavelength_star)
interp_func_cubic =interp1d(real_star_wavelength, real_star_flux, kind='cubic', bounds_error=False, fill_value="extrapolate")
interpolated_star_flux_cubic = interp_func_cubic(wavelength_star)
linear_flux = interpolated_star_flux_linear
quadratic_flux = interpolated_star_flux_quadratic
cubic_flux = interpolated_star_flux_cubic

# === Load the FITS file ===
hdul2 = fits.open("wecfsto_0232_178822nd_23-12-2024.fits")
data2 = hdul2[0].data
header2 = hdul2[0].header

# === Extract exposure time (commonly stored as EXPTIME or EXPOSURE) ===
exptime = header2.get("EXPTIME") or header2.get("EXPOSURE")

if exptime is None:
    raise ValueError("Exposure time not found in FITS header. Check if it's stored under a different keyword.")

# === Convert counts to counts/sec ===
flux_counts = data2  # If this is already the flux array
flux_cps = flux_counts / exptime

# === Optional: Wavelength array (from header or WCS info) ===
# If it's a linear dispersion spectrum
crval2 = header2["CRVAL1"]  # Starting wavelength
cdelt2 = header2["CDELT1"]  # Wavelength increment per pixel
n_pix = len(flux_cps)
wavelength2 = crval2 + cdelt2 * np.arange(n_pix)

# === Plot ===
plt.plot(wavelength2, flux_counts, label="Processed Spectrum (counts)")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux (counts)")
plt.title("Processed Spectrum")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

plt.plot(wavelength2, flux_cps, label="Processed Spectrum (counts/s)")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux (counts/s)")
plt.title("Processed Spectrum")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# STEP 5: Plotting
# Create subplots for four separate plots
fig, axs = plt.subplots(4, 1, figsize=(8, 12))

# Plot 1: Original Flux
axs[0].plot(real_star_wavelength, real_star_flux, label='Original Flux', color='blue')
axs[0].set_title('Original Flux')
axs[0].set_xlabel('Wavelength')
axs[0].set_ylabel('Flux')
axs[0].grid(True)
axs[0].set_xlim(3300,6000)

# Plot 2: 
axs[1].plot(wavelength_star, linear_flux, label='Linear Flux', color='green')
axs[1].set_title('Linear Flux')
axs[1].set_xlabel('Wavelength')
axs[1].set_ylabel('Flux')
axs[1].grid(True)

# Plot 3: Quadratic Flux
axs[2].plot(wavelength_star, quadratic_flux, label='Quadratic Flux', color='red', linestyle='--')
axs[2].set_title('Quadratic Flux')
axs[2].set_xlabel('Wavelength')
axs[2].set_ylabel('Flux')
axs[2].grid(True)

#Plot 4: Cubic Flux 
axs[3].plot(wavelength_star, cubic_flux, label='Cubic Flux', color='orange')
axs[3].set_title('Cubic Flux')
axs[3].set_xlabel('Wavelength')
axs[3].set_ylabel('Flux')
axs[3].grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()
plt.savefig('Extrapolation_Interpolation_wecfsto_232')

# === SAVE TO CSV ===
resampled_df = pd.DataFrame({
    "wavelength": wavelength_star,
    "Linear_flux": linear_flux,
    "Quadratic_flux": quadratic_flux,
    "Cubic_flux": cubic_flux,
    "Counts": flux_counts,
    "Counts/sec": flux_cps
})
resampled_df.to_csv("Star_Spectrum_interpolated_extrapolated_wecfsto_232.csv", index=False)
print("Saved resampled spectrum to 'spectrum_interpolated_extrapolated_wecfsto_232.csv'")