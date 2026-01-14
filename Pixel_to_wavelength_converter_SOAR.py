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
    slope = 1.428038473597098 
    intercept = 2869.633658089779
    return slope*x+intercept

# Replace with your actual .fits file path
fits_file_path = 'wecfzst_0232_178822nd_23-12-2024_445_ws_1.fits'

# Open the .fits file
with fits.open(fits_file_path) as hdul:
    data = hdul[0].data

# Check the shape of the data
print(f"Data shape: {data.shape}")

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
plt.savefig('wecfzst_0232_178822nd_23-12-2024_445_ws_1.png')