#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:50:30 2025

@author: bmr0043
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# === Load the FITS file ===
hdul = fits.open("File_name.fits")
data = hdul[0].data
header = hdul[0].header

# === Extract exposure time (commonly stored as EXPTIME or EXPOSURE) ===
exptime = header.get("EXPTIME") or header.get("EXPOSURE")

if exptime is None:
    raise ValueError("Exposure time not found in FITS header. Check if it's stored under a different keyword.")

# === Convert counts to counts/sec ===
flux_counts = data  # If this is already the flux array
flux_cps = flux_counts / exptime

# === Optional: Wavelength array (from header or WCS info) ===
# If it's a linear dispersion spectrum
crval1 = header["CRVAL1"]  # Starting wavelength
cdelt1 = header["CDELT1"]  # Wavelength increment per pixel
n_pix = len(flux_cps)
wavelength = crval1 + cdelt1 * np.arange(n_pix)

# === Plot ===
plt.plot(wavelength, flux_cps, label="Processed Spectrum (counts/s)")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux (counts/s)")
plt.title("Processed Spectrum")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# === Optional: Save to CSV ===
import pandas as pd

# Save flux counts/sec and flux counts
df_cps = pd.DataFrame({"wavelength": wavelength, "flux_cps": flux_cps, "flux_counts": flux_counts})
df_cps.to_csv("processed_spectrum_cps.csv", index=False)
print("Saved processed spectrum in counts/sec to 'processed_spectrum_cps.csv'")

