#!/bin/python3
"""----------------------------------------------------------------------------
Normalization script for actual data
----------------------------------------------------------------------------"""

# importing needed modules
import numpy as np
import pandas as pd
import os,glob

# obtaining filenames
fnames = sorted(glob.glob1(os.getcwd()+"/../01_numericalSolution/","*.csv"))

# fixing normalized data value range
minVal = 0.05
maxVal = 0.95

# looping through files
for name in fnames:
    # reading data
    fid = pd.read_csv("../01_numericalSolution/"+name)

    # taking a copy of dataframe to store normalized values
    df = fid.copy()

    # defining lists to store min max values
    act_min = []; act_max = []
    mod_min = []; mod_max = []

    # looping through columns
    for column in fid.columns:
        # obtaining actual min and max values
        col_act_min = fid[column].min()
        col_act_max = fid[column].max()

        # computing modified min max values
        col_mod_min = (maxVal*col_act_min - minVal*col_act_max)/(maxVal-minVal)
        col_mod_max = (col_act_min - col_mod_min)/minVal + col_mod_min

        # normalizing data and storing to dataframe
        df[column] = (fid[column] - col_mod_min)/(col_mod_max - col_mod_min)

        # appending min max values to the lists
        act_min.append(col_act_min); act_max.append(col_act_max)
        mod_min.append(col_mod_min); mod_max.append(col_mod_max)

    # preparing dataframe to store min max values
    fid_mm = pd.DataFrame([act_min,act_max,mod_min,mod_max],
                          columns = fid.columns)

    # writing normalized data and minmax values to file
    df.to_csv("normalized_"+name, index = None)
    fid_mm.to_csv("minMax_"+name, index = None)
