#!/bin/python3
"""----------------------------------------------------------------------------
Least Squares estimation of CD using full equation
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
X = np.zeros([N,4])

for idx in range(N):
    i = idx + 1
    dalphadt = (alpha[i+1] - alpha[i-1])/2.0/dt
    Q = 0.5*rho*V[i]**2

    y[idx] = -(dalphadt - q[i] + T/m/V[i]*np.sin(alpha[i]) - g/V[i]*np.cos(alpha[i] - theta[i]))/(Q*S/m/V[i])

    X[idx,0] = 1.0
    X[idx,1] = alpha[i]
    X[idx,2] = q[i]*c_bar/2.0/V[i]
    X[idx,3] = dele[i]

# computing coefficients through matrix computations
mat1 = np.linalg.inv(np.matmul(X.T,X))
coeff = np.matmul(mat1, np.matmul(X.T, y))

CL_0, CL_alpha, CL_q, CL_dele = coeff

print("Coefficients : ")
print("CL_0 = ",CL_0)
print("CL_alpha = ",CL_alpha)
print("CL_q = ",CL_q)
print("CL_dele = ",CL_dele)

print("\n CL_0 error percentage = ",abs(CL_0-0.365)/0.365*100)
print(" CL_alpha error percentage = ",abs(CL_alpha-4.97)/4.97*100)
print(" CL_q error percentage = ",abs(CL_q-37.3)/37.3*100)
print(" CL_dele error percentage = ",abs(CL_dele-0.26)/0.26*100)

# writing to file
fid = open("AD_coefficients.csv", "w")
fid.writelines("Coefficients : "+"\n")
fid.writelines("CL_0 = "+str(CL_0)+"\n")
fid.writelines("CL_alpha = "+str(CL_alpha)+"\n")
fid.writelines("CL_q = "+str(CL_q)+"\n")
fid.writelines("CL_dele = "+str(CL_dele)+"\n")
fid.close()

