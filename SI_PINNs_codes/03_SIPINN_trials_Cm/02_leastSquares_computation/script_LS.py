#!/bin/python3
"""----------------------------------------------------------------------------
Least Squares estimation of Cm using full equation
----------------------------------------------------------------------------"""

# importing needed modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading computed data
fid   = pd.read_csv("../../01_numericalSolution/computed_data.csv")
q     = fid["q"].to_numpy()
dele  = fid["dele"].to_numpy()
alpha = fid["alpha"].to_numpy()
V     = fid["V"].to_numpy()

dt = fid['time'].iloc[1] - fid['time'].iloc[0]

# defning other constant parameters
Iyy   = 907.0
S     = 12.47
c_bar = 1.211
rho   = 1.225

# framing matrices
N = fid.shape[0]-2 #  due to central difference
y = np.zeros(N)
X = np.zeros([N,4])

for idx in range(N):
    i = idx + 1
    dqdt = (q[i+1] - q[i-1])/2.0/dt
    Q = 0.5*rho*V[i]**2

    y[idx] = dqdt*Iyy/Q/S/c_bar

    X[idx,0] = 1.0
    X[idx,1] = alpha[i]
    X[idx,2] = q[i]*c_bar/2.0/V[i]
    X[idx,3] = dele[i]

# computing coefficients through matrix computations
mat1 = np.linalg.inv(np.matmul(X.T,X))
theta = np.matmul(mat1, np.matmul(X.T, y))

Cm_0, Cm_alpha, Cm_q, Cm_dele = theta

print("Coefficients : ")
print("Cm_0 = ",Cm_0)
print("Cm_alpha = ",Cm_alpha)
print("Cm_q = ",Cm_q)
print("Cm_dele = ",Cm_dele)

# writing to file
fid = open("AD_coefficients.csv", "w")
fid.writelines("Coefficients : "+"\n")
fid.writelines("Cm_0 = "+str(Cm_0)+"\n")
fid.writelines("Cm_alpha = "+str(Cm_alpha)+"\n")
fid.writelines("Cm_q = "+str(Cm_q)+"\n")
fid.writelines("Cm_dele = "+str(Cm_dele)+"\n")

