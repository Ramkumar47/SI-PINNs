import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# reading prediction data
fid_cd_f = pd.read_csv("../../01_numericalSolution/computed_data.csv") #  <<== for cd prediction training
fid_cd_p = pd.read_csv("predicted_data.csv")
fid_cd_f = fid_cd_f.iloc[np.arange(0,fid_cd_f.shape[0],4)]

# predicted CD_coeffs
p_CD_0 = 0.03609177851810485
p_CD_alpha = 0.040168739801151855
p_CD_dele = 0.025303332556625723

# actual CD coeffs
a_CD_0 = 0.036
a_CD_alpha = 0.041
a_CD_dele = 0.026

# least squares output
l_CD_0 = 0.037063385835364566
l_CD_alpha = 0.029253225937345192
l_CD_dele = 0.011489827221586419

# computing predicted and actual CD
CD_p = p_CD_0 + p_CD_alpha*fid_cd_f['alpha'] + p_CD_dele*fid_cd_f['dele']
CD_a = a_CD_0 + a_CD_alpha*fid_cd_f['alpha'] + a_CD_dele*fid_cd_f['dele']
CD_l = l_CD_0 + l_CD_alpha*fid_cd_f['alpha'] + l_CD_dele*fid_cd_f['dele']

# plotting predicted vs actual CD graph
#  plt.rcParams.update({'font.size':15})
plt.figure()
plt.plot(fid_cd_f['time'], CD_p,'-b',label='SI-PINN')
#  plt.plot(fid_cd_f['time'], CD_l,'-g',label='LS')
plt.plot(fid_cd_f['time'], CD_a,'-r',label='actual')
plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("CD")
plt.title("CD")
plt.savefig("CD_output.png", dpi =150)

plt.figure()
plt.plot(fid_cd_f['time'],fid_cd_p['V_pred'],'-b',label='SI-PINN',linewidth=4)
plt.plot(fid_cd_f['time'],fid_cd_f['V'],'-y',label='full data')
plt.plot([1.04,1.04],[39,42],'--k')
plt.plot(fid_cd_f['time'].iloc[13:],fid_cd_p['V_pred'].iloc[13:],'+r',label='training data', markevery=3)
plt.legend(bbox_to_anchor=[1.1,0.9],loc='upper left')
plt.ylabel("V")
plt.xlabel("time")
plt.grid()
plt.savefig("V_comparison.png", dpi = 150, bbox_inches ='tight')

plt.show()
