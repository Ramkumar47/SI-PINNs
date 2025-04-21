#!/bin/python3
"""----------------------------------------------------------------------------
Least Squares estimation of CD using full equation
----------------------------------------------------------------------------"""

# importing needed modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading computed data
fid   = pd.read_csv("../00_missingDataProcessing/computed_data_with_missing.csv")
#  fid   = fid.iloc[np.arange(0,fid.shape[0],2)].reset_index(drop=True)
q     = fid["q"].to_numpy()
dele  = fid["dele"].to_numpy()
alpha = fid["alpha"].to_numpy()
V     = fid["V"].to_numpy()
theta = fid["theta"].to_numpy()

dt = fid['time'].iloc[1] - fid['time'].iloc[0]

# indices of split data
I1 = [25,26]
I2 = [50,51]
I3 = [126,127]

# defning other constant parameters
Iyy   = 907.0
S     = 12.47
c_bar = 1.211
rho   = 1.225
m     = 750.0
T     = 1136.0
g     = 9.81

# framing matrices
N = fid.shape[0]-8 #  due to central difference
y = np.zeros(N)
X = np.zeros([N,3])
count = 0

for i in np.arange(1,I1[0]): #  within first part
    dVdt = (V[i+1] - V[i-1])/2.0/dt
    Q = 0.5*rho*V[i]**2
    y[count] = -(dVdt - T/m*np.cos(alpha[i]) - g*np.sin(alpha[i] - theta[i]))*m/Q/S
    X[count,0] = 1.0
    X[count,1] = alpha[i]
    X[count,2] = dele[i]
    count += 1

for i in np.arange(I1[1]+1,I2[0]): #  within second part
    dVdt = (V[i+1] - V[i-1])/2.0/dt
    Q = 0.5*rho*V[i]**2
    y[count] = -(dVdt - T/m*np.cos(alpha[i]) - g*np.sin(alpha[i] - theta[i]))*m/Q/S
    X[count,0] = 1.0
    X[count,1] = alpha[i]
    X[count,2] = dele[i]
    count += 1

for i in np.arange(I2[1]+1,I3[0]): #  within third part
    dVdt = (V[i+1] - V[i-1])/2.0/dt
    Q = 0.5*rho*V[i]**2
    y[count] = -(dVdt - T/m*np.cos(alpha[i]) - g*np.sin(alpha[i] - theta[i]))*m/Q/S
    X[count,0] = 1.0
    X[count,1] = alpha[i]
    X[count,2] = dele[i]
    count += 1

for i in np.arange(I3[1]+1,fid.shape[0]-1): #  within third part
    dVdt = (V[i+1] - V[i-1])/2.0/dt
    Q = 0.5*rho*V[i]**2
    y[count] = -(dVdt - T/m*np.cos(alpha[i]) - g*np.sin(alpha[i] - theta[i]))*m/Q/S
    X[count,0] = 1.0
    X[count,1] = alpha[i]
    X[count,2] = dele[i]
    count += 1

# computing coefficients through matrix computations
mat1 = np.linalg.inv(np.matmul(X.T,X))
coeff = np.matmul(mat1, np.matmul(X.T, y))

CD_0, CD_alpha, CD_dele = coeff

print("Coefficients : ")
print("Cm_0 = ",CD_0)
print("CD_alpha = ",CD_alpha)
print("CD_dele = ",CD_dele)

# writing to file
fileid = open("AD_coefficients.csv", "w")
fileid.writelines("Coefficients : "+"\n")
fileid.writelines("CD_0 = "+str(CD_0)+"\n")
fileid.writelines("CD_alpha = "+str(CD_alpha)+"\n")
fileid.writelines("CD_dele = "+str(CD_dele)+"\n")
fileid.close()

