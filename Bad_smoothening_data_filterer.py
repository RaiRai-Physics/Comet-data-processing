#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 10:15:28 2025

@author: bmr0043
"""

import pandas as pd

# Load your original CSV
df = pd.read_csv("Smoothed_Flux_to_counts_per_sec_wecfzst_232_with_diff_ratio_2.csv")

# Define the related column index mapping (indexing starts at 0)
related_cols = {7: 2, 8: 3, 9: 4, 10: 5, 11: 6}  # col 8->3, 9->4, ...

# Copy the DataFrame to work on
filtered_df = df.copy()

# Create dictionaries to store counts
good_counts = {}
bad_counts = {}

# Loop over each target column (8 to 12, i.e., index 7 to 11)
for col_idx in range(7, 12):
    col_name = df.columns[col_idx]
    related_col_name = df.columns[related_cols[col_idx]]

    # Take absolute values
    abs_vals = df[col_name].abs()

    # Check which values are > 5
    mask_bad = abs_vals > 5

    # Count good and bad
    good = (abs_vals <= 5).sum()
    bad = (abs_vals > 5).sum()

    # Store counts
    good_counts[col_name] = good
    bad_counts[col_name] = bad

    # Replace "bad" entries with 'Bad' in both the target and related columns
    filtered_df.loc[mask_bad, col_name] = 'Bad'
    filtered_df.loc[mask_bad, related_col_name] = 'Bad'

# Save the new filtered CSV
filtered_df.to_csv("filtered_Smoothening_output_counts_per_sec.csv", index=False)

# Print summary
print("===== Good/Bad Counts per Column (Cols 8–12) =====")
for col_name in good_counts:
    print(f"{col_name} → Good (≤5): {good_counts[col_name]}, Bad (>5): {bad_counts[col_name]}")