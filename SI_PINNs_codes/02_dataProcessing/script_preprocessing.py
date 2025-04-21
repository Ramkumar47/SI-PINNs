#!/bin/python3
"""----------------------------------------------------------------------------
preprocessing the selected computed dataset
for the discrete PINN model
----------------------------------------------------------------------------"""

# importing needed modules
import numpy as np
import pandas as pd

# reading dataset
fid = pd.read_csv("../01_numericalSolution/computed_data.csv",
                  usecols = ["time","V","alpha","q","dele","CD","CL","Cm"])
df = fid.copy()

# lists to store actual min max and modified min max values
act_min = []
act_max = []
mod_min = []
mod_max = []

# normalizing dataset
for column in fid.columns:
    # obtaining min and max values
    minVal = fid.min()[column]
    maxVal = fid.max()[column]

    # obtaining 10% of the data range for extension
    fraction = 0.1
    ext = abs(maxVal - minVal)*fraction

    if ext == 0:
        minVal = 0
        maxVal = maxVal*1.0
        ext = maxVal*fraction

    # computing modified min max values
    modMinVal = minVal - ext
    modMaxVal = maxVal + ext

    # normalizing data
    df[column] = (fid[column] - modMinVal)/(modMaxVal - modMinVal)

    # appending values to the list
    act_min.append(minVal)
    act_max.append(maxVal)
    mod_min.append(modMinVal)
    mod_max.append(modMaxVal)

# writing normalized data to file
df.to_csv("normalized_data.csv", index = None)

# preparing dataframe for min max values and storing to the file
fid = pd.DataFrame([act_min,act_max,mod_min,mod_max],
                   columns = fid.columns)
fid.to_csv("minMaxValues.csv", index = None)
