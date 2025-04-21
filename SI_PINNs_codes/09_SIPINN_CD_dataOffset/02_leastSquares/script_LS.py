#!/bin/python3
"""----------------------------------------------------------------------------
Least Squares estimation of CD using full equation
----------------------------------------------------------------------------"""

# importing needed modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
comments:
    1 - dropping first 1 second data as one of state variable data is unavailble
        in that time
"""

# reading computed data
fid = pd.read_csv("../../01_numericalSolution/computed_data.csv")
fid   = fid.iloc[np.arange(0,fid.shape[0],4)].reset_index(drop=True)
fid   = fid.iloc[13:].reset_index(drop = True) #  << - 1
q     = fid["q"].to_numpy()
dele  = fid["dele"].to_numpy()
alpha = fid["alpha"].to_numpy()
V     = fid["V"].to_numpy()
theta = fid["theta"].to_numpy()

dt = fid['time'].iloc[1] - fid['time'].iloc[0]

# defning other constant parameters
Iyy   = 907.0
S     = 12.47
c_bar = 1.211
rho   = 1.225
m     = 750.0
T     = 1136.0
g     = 9.81

# framing matrices
N = fid.shape[0]-2 #  due to central difference
y = np.zeros(N)
X = np.zeros([N,3])

for idx in range(N):
    i = idx + 1
    dVdt = (V[i+1] - V[i-1])/2.0/dt
    Q = 0.5*rho*V[i]**2

    y[idx] = -(dVdt - T/m*np.cos(alpha[i]) - g*np.sin(alpha[i] - theta[i]))*m/Q/S

    X[idx,0] = 1.0
    X[idx,1] = alpha[i]
    X[idx,2] = dele[i]

# computing coefficients through matrix computations
mat1 = np.linalg.inv(np.matmul(X.T,X))
coeff = np.matmul(mat1, np.matmul(X.T, y))

CD_0, CD_alpha, CD_dele = coeff

print("Coefficients : ")
print("Cm_0 = ",CD_0)
print("CD_alpha = ",CD_alpha)
print("CD_dele = ",CD_dele)

# writing to file
fid = open("AD_coefficients.csv", "w")
fid.writelines("Coefficients : "+"\n")
fid.writelines("CD_0 = "+str(CD_0)+"\n")
fid.writelines("CD_alpha = "+str(CD_alpha)+"\n")
fid.writelines("CD_dele = "+str(CD_dele)+"\n")
fid.close()

