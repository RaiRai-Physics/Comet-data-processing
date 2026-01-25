#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:17:11 2025

@author: bmr0043
"""

from astropy.io import fits
import astroscrappy
import matplotlib.pyplot as plt

# Load star
raw_file = '0232_178822nd_23-12-2024.fits'  # replace with your actual path
with fits.open(raw_file) as hdul:
    raw_data = hdul[0].data
    raw_header = hdul[0].header

# Load lamp
arc_file = '0229_178822nd_23-12-2024.fits'  # replace with your actual path
with fits.open(arc_file) as hdul:
    arc_data = hdul[0].data

# Cosmic ray removal on raw image
cr_mask_obj, cleaned_raw = astroscrappy.detect_cosmics(raw_data,
                                                        sigclip=4.5,
                                                        sigfrac=0.3,
                                                        objlim=5.0)

# Cosmic ray removal on arc image (optional but recommended)
cr_mask_arc, cleaned_arc = astroscrappy.detect_cosmics(arc_data,
                                                        sigclip=4.5,
                                                        sigfrac=0.3,
                                                        objlim=5.0)

# Subtract arc from cleaned object
final_data = cleaned_raw - cleaned_arc

# Save final cosmic-ray-cleaned and lamp-subtracted image
final_fits = '0232_178822nd_23-12-2024_cleaned_lamp_subtracted.fits'
fits.writeto(final_fits, final_data, raw_header, overwrite=True)

# Optional visualization
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(raw_data, origin='lower', cmap='gray')
ax[0].set_title("Original Raw")
ax[1].imshow(cleaned_raw, origin='lower', cmap='gray')
ax[1].set_title("Cleaned Raw (CR removed)")
ax[2].imshow(final_data, origin='lower', cmap='gray')
ax[2].set_title("Final (CR removed & Arc subtracted)")
plt.tight_layout()
plt.show()
