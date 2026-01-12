#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 08:04:10 2025

@author: bmr0043
"""

from astropy.io import fits
import matplotlib.pyplot as plt

# Load the raw science spectrum
science_hdul = fits.open('File_name.fits')
science_data = science_hdul[0].data  # 1D spectrum


# Plot the raw spectrum
plt.figure(figsize=(8, 5))
plt.plot(science_data, label='Raw Science Spectrum')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.title('Plotting of 1D raw Spectrum')
plt.legend()
plt.show()
plt.savefig('cometo')