#!/bin/python3
"""----------------------------------------------------------------------------
missing data preparation script

only cut data was prepared
with same normalization levels as the overall dataset
----------------------------------------------------------------------------"""

# importing needed modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# reading normalized data
fid = pd.read_csv("../../01_numericalSolution/computed_data.csv")
fid_norm = pd.read_csv("../../02_dataProcessing/normalized_computed_data.csv")

# preparing needed directories
os.system("rm -rf images && mkdir images")
os.system("rm -rf dataset && mkdir dataset")

#  # plotting data
#
#  plt.figure()
#  plt.plot(fid["time"],fid["V"],'-b')
#  plt.xlabel("time")
#  plt.ylabel("V")
#  plt.title("V")
#  plt.grid()
#
#  plt.figure()
#  plt.plot(fid["time"],fid["alpha"],'-b')
#  plt.xlabel("time")
#  plt.ylabel("alpha")
#  plt.title("alpha")
#  plt.grid()
#
#  plt.figure()
#  plt.plot(fid["time"],fid["theta"],'-b')
#  plt.xlabel("time")
#  plt.ylabel("theta")
#  plt.title("theta")
#  plt.grid()
#
#  plt.figure()
#  plt.plot(fid["time"],fid["dele"],'-b')
#  plt.xlabel("time")
#  plt.ylabel("dele")
#  plt.title("dele")
#  plt.grid()
#
#  plt.show()

# separating dataframe
fid_V     = fid[["time","V"]]
fid_alpha = fid[["time","alpha"]]
fid_theta = fid[["time","theta"]]
fid_dele  = fid[["time","dele"]]
fid_V_norm     = fid_norm[["time","V"]]
fid_alpha_norm = fid_norm[["time","alpha"]]
fid_theta_norm = fid_norm[["time","theta"]]
fid_dele_norm  = fid_norm[["time","dele"]]

# removing data segments

V_idx1 = list(fid_V.loc[(fid_V["time"] > 2) & (fid_V["time"] < 3)].index)
V_idx2 = list(fid_V.loc[(fid_V["time"] > 5) & (fid_V["time"] < 6)].index)
V_idx3 = list(fid_V.loc[(fid_V["time"] > 9) & (fid_V["time"] < 11)].index)
fid_V = fid_V.drop(V_idx1+V_idx2+V_idx3).reset_index(drop=True)
fid_V_norm = fid_V_norm.drop(V_idx1+V_idx2+V_idx3).reset_index(drop=True)

alpha_idx1 = list(fid_alpha.loc[(fid_alpha["time"] > 1) & (fid_alpha["time"] < 2)].index)
alpha_idx2 = list(fid_alpha.loc[(fid_alpha["time"] > 6) & (fid_alpha["time"] < 7)].index)
alpha_idx3 = list(fid_alpha.loc[(fid_alpha["time"] > 11) & (fid_alpha["time"] < 12)].index)
fid_alpha = fid_alpha.drop(alpha_idx1+alpha_idx2+alpha_idx3).reset_index(drop=True)
fid_alpha_norm = fid_alpha_norm.drop(alpha_idx1+alpha_idx2+alpha_idx3).reset_index(drop=True)

theta_idx1 = list(fid_theta.loc[(fid_theta["time"] > 1) & (fid_theta["time"] < 2)].index)
theta_idx2 = list(fid_theta.loc[(fid_theta["time"] > 10) & (fid_theta["time"] < 11)].index)
theta_idx3 = list(fid_theta.loc[(fid_theta["time"] > 13) & (fid_theta["time"] < 14)].index)
fid_theta = fid_theta.drop(theta_idx1+theta_idx2+theta_idx3).reset_index(drop=True)
fid_theta_norm = fid_theta_norm.drop(theta_idx1+theta_idx2+theta_idx3).reset_index(drop=True)

dele_idx1 = list(fid_dele.loc[(fid_dele["time"] > 3) & (fid_dele["time"] < 4)].index)
dele_idx2 = list(fid_dele.loc[(fid_dele["time"] > 7) & (fid_dele["time"] < 8)].index)
dele_idx3 = list(fid_dele.loc[(fid_dele["time"] > 12) & (fid_dele["time"] < 13)].index)
fid_dele = fid_dele.drop(dele_idx1+dele_idx2+dele_idx3).reset_index(drop=True)
fid_dele_norm = fid_dele_norm.drop(dele_idx1+dele_idx2+dele_idx3).reset_index(drop=True)



# plotting graphs

plt.figure()
plt.plot(fid["time"],fid["V"],'-b',label="full")
plt.plot(fid_V["time"],fid_V["V"],'.r',label="segment")
plt.legend()
plt.grid()
plt.xlabel("time")
plt.ylabel("V")
plt.title("V")
plt.savefig("images/V.png")

plt.figure()
plt.plot(fid["time"],fid["alpha"],'-b',label="full")
plt.plot(fid_alpha["time"],fid_alpha["alpha"],'.r',label="segment")
plt.legend()
plt.grid()
plt.xlabel("time")
plt.ylabel("alpha")
plt.title("alpha")
plt.savefig("images/alpha.png")

plt.figure()
plt.plot(fid["time"],fid["theta"],'-b',label="full")
plt.plot(fid_theta["time"],fid_theta["theta"],'.r',label="segment")
plt.legend()
plt.grid()
plt.xlabel("time")
plt.ylabel("theta")
plt.title("theta")
plt.savefig("images/theta.png")

plt.figure()
plt.plot(fid["time"],fid["dele"],'-b',label="full")
plt.plot(fid_dele["time"],fid_dele["dele"],'.r',label="segment")
plt.legend()
plt.grid()
plt.xlabel("time")
plt.ylabel("dele")
plt.title("dele")
plt.savefig("images/dele.png")

#  plt.figure()
#  plt.plot(fid_norm["time"],fid_norm["V"],'-b',label="full")
#  plt.plot(fid_V_norm["time"],fid_V_norm["V"],'.r',label="segment")
#  plt.legend()
#  plt.grid()
#  plt.xlabel("time")
#  plt.ylabel("V")
#  plt.title("V")
#  plt.savefig("images/V_norm.png")
#
#  plt.figure()
#  plt.plot(fid_norm["time"],fid_norm["alpha"],'-b',label="full")
#  plt.plot(fid_alpha_norm["time"],fid_alpha_norm["alpha"],'.r',label="segment")
#  plt.legend()
#  plt.grid()
#  plt.xlabel("time")
#  plt.ylabel("alpha")
#  plt.title("alpha")
#  plt.savefig("images/alpha_norm.png")
#
#  plt.figure()
#  plt.plot(fid_norm["time"],fid_norm["theta"],'-b',label="full")
#  plt.plot(fid_theta_norm["time"],fid_theta_norm["theta"],'.r',label="segment")
#  plt.legend()
#  plt.grid()
#  plt.xlabel("time")
#  plt.ylabel("theta")
#  plt.title("theta")
#  plt.savefig("images/theta_norm.png")
#
#  plt.figure()
#  plt.plot(fid_norm["time"],fid_norm["dele"],'-b',label="full")
#  plt.plot(fid_dele_norm["time"],fid_dele_norm["dele"],'.r',label="segment")
#  plt.legend()
#  plt.grid()
#  plt.xlabel("time")
#  plt.ylabel("dele")
#  plt.title("dele")
#  plt.savefig("images/dele_norm.png")

plt.show()


# saving cut datasets
fid_V_norm.to_csv("dataset/V_dataset.csv", index = None)
fid_alpha_norm.to_csv("dataset/alpha_dataset.csv", index = None)
fid_theta_norm.to_csv("dataset/theta_dataset.csv", index = None)
fid_dele_norm.to_csv("dataset/dele_dataset.csv", index = None)

print("done")
