#!/bin/python3
"""----------------------------------------------------------------------------
System Identification PINN (SIPINN) Program
----------------------------------------------------------------------------"""

# hiding system warnings #  needs to be on top for complete suppression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = "FALSE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

#  from functions import build_dense_layer
from customModel import CustomModel

from copy import copy as cp #  to prevent absolute referencing

tf.keras.backend.set_floatx('float64')

# hiding unnecessary warnings--------------------------------------------------
# tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# disabling eager execution
#  tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution() #  needed for custom training loop

tf.compat.v1.experimental.output_all_intermediates(True)

# ML parameters definition-----------------------------------------------------
epochs = 100000
learning_rate = 1e-3
kern_init = tf.keras.initializers.GlorotNormal(seed = 1)
bias_init = tf.keras.initializers.GlorotNormal(seed = 1)
optimizerFunc = tf.keras.optimizers.legacy.Adam(
                        learning_rate=learning_rate)

#  regularizer = tf.keras.regularizers.l1(0)
regularizer = None

# model build------------------------------------------------------------------

model = CustomModel(kernel_initializer=kern_init, bias_initializer=bias_init)

# dataset preparation----------------------------------------------------------
# reading dataset
fid_cd_f = pd.read_csv("../../01_numericalSolution/computed_data.csv") #  <<== for cd prediction training
fid_nd_f = pd.read_csv("../../02_dataProcessing/normalized_data.csv")
fid_cd = pd.read_csv("../00_missingDataProcessing/computed_data_with_missing.csv") #  <<== for missing V training
fid_nd = pd.read_csv("../00_missingDataProcessing/normalized_data_with_missing.csv")

fid_cd_f = fid_cd_f.iloc[np.arange(0,fid_cd_f.shape[0],4)].reset_index(drop=True)
fid_nd_f = fid_nd_f.iloc[np.arange(0,fid_nd_f.shape[0],4)].reset_index(drop=True)
#  fid_cd = fid_cd.iloc[np.arange(0,fid_cd.shape[0],2)].reset_index(drop=True)
#  fid_nd = fid_nd.iloc[np.arange(0,fid_nd.shape[0],2)].reset_index(drop=True)

# getting computed fields to be used in PI function
alpha = fid_cd_f['alpha'].to_numpy()[:,None]
theta = fid_cd_f['theta'].to_numpy()[:,None]
dele = fid_cd_f['dele'].to_numpy()[:,None]

# getting normalized data to be fed in the model for coeff. estimation
V = fid_nd['V'].to_numpy()[:,None]
time = fid_nd['time'].to_numpy()[:,None]
time_pred = fid_nd_f['time'].to_numpy()[:,None]

# getting ad coefficients for output comparison
CD = fid_nd_f['CD'].to_numpy()[:,None]


# training model---------------------------------------------------------------

# custom loss function
def lossFunc(y_true,y_pred):
    lossVal = K.mean(K.square(y_true - y_pred))
    return lossVal

# training function
@tf.function
def training_function():

    # evaluating model for data output
    V_pred, CD_pred = model.predict_x([time])
    mse_data        = lossFunc(V_pred, V)

    # evaluating model for PINN CD output
    res_V,res_coeff = model.predict_f([time_pred,theta,alpha,dele])
    res_V_loss      = lossFunc(res_V*K.constant(0.0),res_V)
    res_coeff_loss  = lossFunc(res_coeff*K.constant(0.0),res_coeff)
    mse_phy         = res_V_loss + res_coeff_loss
    #  mse_phy         = res_V_loss*1.0 #  <==============================================

    # combining physics and data losses
    loss_value = mse_phy + mse_data
    #  loss_value = mse_data*1.0

    # computing gradients of loss w.r.t. model weights
    grads = K.gradients(loss_value, model.trainable_weights)

    # backpropagation
    optimizerFunc.apply_gradients(zip(grads, model.trainable_weights))

    return mse_data, mse_phy

# loading previously trained weights
model.load_weights("./model_weights/weights")

# custom trainning loop
for epoch in range(epochs):

    # training model
    mse_data, mse_phy = training_function()

    # getting coefficients
    CD_0 = model.CD_0.numpy()
    CD_alpha = model.CD_alpha.numpy()
    CD_dele = model.CD_dele.numpy()

    print("epoch : ",epoch+1,
          "\t mse_data : ", np.round(mse_data.numpy(),8),
          " mse_phy : ", np.round(mse_phy.numpy(),8))

# saving weights
model.save_weights(os.getcwd()+"/model_weights/weights")

# prediction-------------------------------------------------------------------
V_pred,CD_pred = model.predict_x([time_pred])

V_res,coeff_res = model.predict_f([time_pred,theta,alpha,dele])

V_res = abs(V_res)
coeff_res = abs(coeff_res)

# scaling up CD value
fid_MM = pd.read_csv("../../02_dataProcessing/minMaxValues.csv")
CD_min = fid_MM['CD'].iloc[2]
CD_max = fid_MM['CD'].iloc[3]
V_min = fid_MM['V'].iloc[2]
V_max = fid_MM['V'].iloc[3]
CD_pred_s = CD_pred*(CD_max-CD_min) + CD_min
CD_s = CD*(CD_max-CD_min) + CD_min

# error computation
CD_error = np.abs(CD_pred - CD)/np.max(CD)*100.0

plt.rcParams.update({'font.size':15})
#  plt.figure(figsize=(16,8))
plt.figure()
#  plt.plot(CD_pred_s,'-b', label = "predicted")
#  plt.plot(CD_s,'-r', label = "actual")
plt.plot(time_pred,CD_pred,'-b', label = "predicted")
plt.plot(time_pred,CD,'-r', label = "actual")
plt.grid()
plt.xlabel("data count")
plt.ylabel("CD")
plt.legend()
plt.title("CD")
plt.savefig("CD.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(time_pred,V_pred,'-b',label = "pred")
plt.plot(time,V,'-r',label = "actual")
plt.grid()
plt.legend()
plt.title("V")
plt.xlabel("non-dimensional time")
plt.ylabel("V")
plt.savefig("V.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(V_res,'-b',label = "V")
plt.plot(coeff_res,'-r',label = "coeff. eqn")
plt.grid()
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.1,0.9))
plt.title("residual")
plt.xlabel("data count")
plt.ylabel("residual value")
plt.savefig("residual.png", dpi = 150, bbox_inches = "tight")

plt.show()
#  plt.close()

print("\n")
print("CD Error :")
print("\t max = ",CD_error.max())
print("\t min = ",CD_error.min())
print("\t avg = ",CD_error.mean())


# printing model trainable parameters
print("\nModel computed A/D coefficients : ")
#  print("CL_0     = ", model.CL_0.numpy())
#  print("CL_alpha = ", model.CL_alpha.numpy())
#  print("CL_q     = ", model.CL_q.numpy())
#  print("CL_dele  = ", model.CL_dele.numpy())
print("CD_0     = ", model.CD_0.numpy())
print("CD_alpha = ", model.CD_alpha.numpy())
print("CD_dele  = ", model.CD_dele.numpy())
#  print("Cm_0     = ", model.Cm_0.numpy())
#  print("Cm_alpha = ", model.Cm_alpha.numpy())
#  print("Cm_q     = ", model.Cm_q.numpy())
#  print("Cm_dele  = ", model.Cm_dele.numpy())

# writing predicted and actual values to the file
fid = pd.DataFrame(np.transpose([CD_s.flatten(),CD_pred_s.numpy().flatten()]),
                   columns = ["CD_act","CD_pred"])
fid['Error'] = abs(fid['CD_act'] - fid['CD_pred'])/fid['CD_act']*100.0
fid['V_pred'] = V_pred*(V_max-V_min)+V_min
fid.to_csv(os.getcwd()+"/predicted_data.csv", index = None)

# writing computed AD derivative values to file
fid = open(os.getcwd()+"/AD_derivatives.csv", "w")
fid.writelines("\nModel computed A/D coefficients : ")
fid.writelines("\nCL_0     = "+str(model.CL_0.numpy()))
fid.writelines("\nCL_alpha = "+str(model.CL_alpha.numpy()))
fid.writelines("\nCL_q     = "+str(model.CL_q.numpy()))
fid.writelines("\nCL_dele  = "+str(model.CL_dele.numpy()))
fid.writelines("\nCD_0     = "+str(model.CD_0.numpy()))
fid.writelines("\nCD_alpha = "+str(model.CD_alpha.numpy()))
fid.writelines("\nCD_dele  = "+str(model.CD_dele.numpy()))
fid.writelines("\nCm_0     = "+str(model.Cm_0.numpy()))
fid.writelines("\nCm_alpha = "+str(model.Cm_alpha.numpy()))
fid.writelines("\nCm_q     = "+str(model.Cm_q.numpy()))
fid.writelines("\nCm_dele  = "+str(model.Cm_dele.numpy()))

fid.close()

print("\n done")
