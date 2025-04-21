#!/bin/python3
"""----------------------------------------------------------------------------
missing data generation
----------------------------------------------------------------------------"""

# importing needed modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading full data
fid_cd = pd.read_csv("../../01_numericalSolution/computed_data.csv")
fid_nd = pd.read_csv("../../02_dataProcessing/normalized_data.csv")

# filtering data to 370 pnts
fid_cd = fid_cd.iloc[np.arange(0,fid_cd.shape[0],4)].reset_index(drop=True)
fid_nd = fid_nd.iloc[np.arange(0,fid_nd.shape[0],4)].reset_index(drop=True)

# missing data points in range : 5 - 6
fid_cd_m1 = fid_cd[fid_cd["time"] < 6.0]
drop_index = fid_cd_m1[fid_cd_m1["time"] > 5.0].index
fid_cd_m1 = fid_cd.drop(drop_index)
fid_nd_m1 = fid_nd.drop(drop_index)

# missing data points in range : 12 - 13
fid_cd_m2 = fid_cd_m1[fid_cd_m1["time"] < 13.0]
drop_index = fid_cd_m2[fid_cd_m2["time"] > 12.0].index
fid_cd_m2 = fid_cd_m1.drop(drop_index)
fid_nd_m2 = fid_nd_m1.drop(drop_index)

# missing data points in range : 2 - 3
fid_cd_m3 = fid_cd_m2[fid_cd_m2["time"] < 3.0]
drop_index = fid_cd_m3[fid_cd_m3["time"] > 2.0].index
fid_cd_m3 = fid_cd_m2.drop(drop_index)
fid_nd_m3 = fid_nd_m2.drop(drop_index)

fid_cd_m3 = fid_cd_m3.reset_index(drop = True)
fid_nd_m3 = fid_nd_m3.reset_index(drop = True)

# saving missing dataset
fid_cd_m3.to_csv("computed_data_with_missing.csv", index = None)
fid_nd_m3.to_csv("normalized_data_with_missing.csv", index = None)

# visualizing data
plt.figure()
plt.plot(fid_cd["time"], fid_cd["V"],'-b',label = "actual data")
plt.plot(fid_cd_m3["time"], fid_cd_m3["V"],'.r',label = "data with missing")
plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("V")
plt.savefig("missed_data_output_V.png", dpi = 150)

plt.show()
