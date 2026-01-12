#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 07:47:33 2025

@author: bmr0043
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the 2D FITS file
arc_hdul = fits.open('File_name.fits')  # Replace with your FITS file
arc_spectrum = arc_hdul[0].data  # Assuming the spectrum is in the first HDU
# Extract a 1D spectrum by summing along the spatial axis
science_1d_spectrum = np.sum(arc_spectrum, axis=0)
arc_hdul.close()

# Detect emission peaks 
arc_peaks, _ = find_peaks(science_1d_spectrum, height=500)  # Adjust height threshold as needed

# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(science_1d_spectrum, label="Arc Lamp Spectrum", color='blue')
plt.scatter(arc_peaks, science_1d_spectrum[arc_peaks], color='red', marker='o', label="Detected Peaks")  # Mark peaks
print(arc_peaks)
# Labels and title
plt.xlabel("Pixel")
plt.ylabel("Intensity")
plt.title("Pixels Versus Intensity")
plt.legend()
plt.show()

