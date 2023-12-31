import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
import os

def fold_massspec(mz_min, mz_max, monoisotopicweight, data):

    # Filter data within the mz range
    data = data[(data['m/z'] >= mz_min) & (data['m/z'] <= mz_max)]

    # Normalize m/z values
    data['m/z'] = (data['m/z'] - mz_min) / monoisotopicweight

    # Interpolation
    x = data['m/z'].values
    y = data['intensity'].values
    f = interp1d(x, y, bounds_error=False, fill_value=0)

    # Create a range of x values
    max_val = x[-1]
    new_x_0_1 = np.linspace(0, 1, 10001)
    step_size = new_x_0_1[1] - new_x_0_1[0]
    new_x_rest = np.arange(1 + step_size, max_val, step_size)
    new_x = np.concatenate((new_x_0_1, new_x_rest))

    # Interpolate y values
    new_y = f(new_x)

    # Fold data in chunks of 10,000 values
    new_y_folded = new_y[:10000].copy()
    for i in range(10000, len(new_y), 10000):
        end_idx = i + 10000 if i + 10000 < len(new_y) else len(new_y)
        chunk = new_y[i:end_idx]
        new_y_folded[:len(chunk)] += chunk

    # Create a DataFrame for the folded data
    folded_data = pd.DataFrame({
        'm/z': new_x[:10000] * monoisotopicweight,
        'intensity': new_y_folded
    })

    # Normalize intensity values
    folded_data['intensity'] = folded_data['intensity'] / folded_data['intensity'].max()

    return folded_data
