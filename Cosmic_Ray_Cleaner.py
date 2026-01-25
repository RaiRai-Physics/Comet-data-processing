#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 07:38:06 2025

@author: bmr0043
"""

from astropy.io import fits
import astroscrappy
import matplotlib.pyplot as plt

# Load your raw spectroscopy image (FITS file)
fits_file = '0232_178822nd_23-12-2024.fits'  # replace with your actual file path
with fits.open(fits_file) as hdul:
    raw_data = hdul[0].data
    header = hdul[0].header

# Run the cosmic ray detection algorithm
cr_mask, cleaned_data = astroscrappy.detect_cosmics(raw_data,
                                                     sigclip=3.0,
                                                     sigfrac=0.3,
                                                     objlim=5.0)

# Save the cleaned image
cleaned_fits = '0232_178822nd_23-12-2024_cleaned_spectrum_3.0.fits'
fits.writeto(cleaned_fits, cleaned_data, header, overwrite=True)

# Optional: visualize original vs cleaned
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(raw_data, origin='lower', cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(cleaned_data, origin='lower', cmap='gray')
ax[1].set_title("Cleaned Image")
plt.tight_layout()
plt.show()
