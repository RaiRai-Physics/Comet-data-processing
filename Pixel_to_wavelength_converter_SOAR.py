#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:29:17 2024

@author: bmr0043
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
def pixel_to_lambda(x):
    slope = 'Enter the slope'
    intercept = 'Enter the intercept'
    return slope*x+intercept

# Replace with your actual .fits file path
fits_file_path = 'File_name.fits'

# Open the .fits file
with fits.open(fits_file_path) as hdul:
    data = hdul[0].data

spectrum_width_pixels = len(data)
pixel_values = np.arange(start=0,stop=spectrum_width_pixels,step=1)
print(f"{pixel_values=}")
lambda_values = pixel_to_lambda(pixel_values)
print(f"{lambda_values=}")
# Plotting the data
plt.figure(figsize=(20, 12))
#plt.plot(data)
plt.plot(lambda_values, data)
plt.title('FITS Data Line Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
plt.savefig('Figure.png')
