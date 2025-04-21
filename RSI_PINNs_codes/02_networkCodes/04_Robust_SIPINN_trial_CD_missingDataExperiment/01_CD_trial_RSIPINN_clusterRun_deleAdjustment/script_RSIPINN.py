#!/bin/python3
"""----------------------------------------------------------------------------
Robust System Identification PINN (RSI-PINN) Program
Estimation of CD
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
fid_nd     = pd.read_csv("../../02_dataProcessing/normalized_computed_data.csv")
CD         = fid_nd['CD'].to_numpy()[:,None]
time       = fid_nd["time"].to_numpy()[:,None]

#  CD         = fid_nd['CD'].to_numpy()[0:-1:4]
#  time       = fid_nd["time"].to_numpy()[0:-1:4]
#  time       = time[:,None]
#  CD         = CD[:,None]

V_full     = fid_nd["V"].to_numpy()[:,None]
alpha_full = fid_nd["alpha"].to_numpy()[:,None]
theta_full = fid_nd["theta"].to_numpy()[:,None]
dele_full  = fid_nd["dele"].to_numpy()[:,None]

fid_nd_V = pd.read_csv("../00_missingDataPreparation/dataset/V_dataset.csv")
time_V   = fid_nd_V["time"].to_numpy()[:,None]
V        = fid_nd_V["V"].to_numpy()[:,None]

fid_nd_alpha = pd.read_csv("../00_missingDataPreparation/dataset/alpha_dataset.csv")
time_alpha   = fid_nd_alpha["time"].to_numpy()[:,None]
alpha        = fid_nd_alpha["alpha"].to_numpy()[:,None]

fid_nd_theta = pd.read_csv("../00_missingDataPreparation/dataset/theta_dataset.csv")
time_theta   = fid_nd_theta["time"].to_numpy()[:,None]
theta        = fid_nd_theta["theta"].to_numpy()[:,None]

fid_nd_dele = pd.read_csv("../00_missingDataPreparation/dataset/dele_dataset.csv")
time_dele   = fid_nd_dele["time"].to_numpy()[:,None]
dele        = fid_nd_dele["dele"].to_numpy()[:,None]

#  # getting computed fields to be used in PI function
#  theta = fid_cd['theta'].to_numpy()[:,None]

#  # getting normalized data to be fed in the model for coeff. estimation
#  alpha = fid_nd['alpha'].to_numpy()[:,None]
#  theta = fid_nd['theta'].to_numpy()[:,None]
#  dele  = fid_nd['dele'].to_numpy()[:,None]
#  V     = fid_nd['V'].to_numpy()[:,None]
#  time  = fid_nd['time'].to_numpy()[:,None]

# training model---------------------------------------------------------------

# defining loss function
lossFunc = tf.keras.losses.MeanSquaredError()

# training function
@tf.function
def training_function():

    #  # evaluating model for data output
    #  V_pred,alpha_pred,theta_pred,dele_pred,CD_pred = model.predict_x([time])
    #  mse_data_V     = lossFunc(V_pred, V)
    #  mse_data_alpha = lossFunc(alpha_pred, alpha)
    #  mse_data_theta = lossFunc(theta_pred, theta)
    #  mse_data_dele  = lossFunc(dele_pred, dele)
    #
    #  # evaluating model for PINN CD output
    #  res_V,res_coeff = model.predict_f([time])
    #  res_V_loss      = lossFunc(res_V*K.constant(0.0),res_V)
    #  res_coeff_loss  = lossFunc(res_coeff*K.constant(0.0),res_coeff)
    #  mse_phy         = res_V_loss + res_coeff_loss
    #
    # combining physics and data losses
    #  loss_value = mse_data_V + mse_data_alpha + mse_data_theta + mse_data_dele
    #  loss_value = mse_phy + mse_data_V + mse_data_alpha + mse_data_theta + mse_data_dele
    #  loss_value = mse_data_V*1.0
    #  loss_value = mse_data_alpha*1.0
    #  loss_value = mse_data_theta*1.0
    #  loss_value = mse_data_dele*1.0

    #  # evaluating model for V data
    #  V_pred,alpha_pred,theta_pred,dele_pred,CD_pred = model.predict_x([time_V])
    #  mse_data = lossFunc(V_pred, V)

    #  # evaluating model for alpha data
    #  V_pred,alpha_pred,theta_pred,dele_pred,CD_pred = model.predict_x([time_alpha])
    #  mse_data = lossFunc(alpha_pred, alpha)

    #  # evaluating model for theta data
    #  V_pred,alpha_pred,theta_pred,dele_pred,CD_pred = model.predict_x([time_theta])
    #  mse_data = lossFunc(theta_pred, theta)

    #  # evaluating model for dele data
    #  V_pred,alpha_pred,theta_pred,dele_pred,CD_pred = model.predict_x([time_dele])
    #  mse_data = lossFunc(dele_pred, dele)

    #  # evaluating model for full data
    #  V_pred,alpha_pred,theta_pred,dele_pred,CD_pred = model.predict_x([time])
    #  mse_V     = lossFunc(V_pred, V_full)
    #  mse_alpha = lossFunc(alpha_pred, alpha_full)
    #  mse_theta = lossFunc(theta_pred, theta_full)
    #  mse_dele  = lossFunc(dele_pred, dele_full)
    #  mse_data  = mse_V + mse_alpha + mse_theta + mse_dele

    mse_data = 1.0

    # evaluating model for PINN CD output
    res_V,res_coeff = model.predict_f([time])
    res_V_loss      = lossFunc(res_V*K.constant(0.0),res_V)
    res_coeff_loss  = lossFunc(res_coeff*K.constant(0.0),res_coeff)
    mse_phy         = res_V_loss + res_coeff_loss
    #  mse_phy = 1.0

    #  loss_value = mse_data*1.0
    #  loss_value = mse_data + mse_phy
    loss_value =  mse_phy*1.0

    # computing gradients of loss w.r.t. model weights
    grads = K.gradients(loss_value, model.trainable_weights)

    # backpropagation
    optimizerFunc.apply_gradients(zip(grads, model.trainable_weights))

    #  return mse_data_V,mse_data_alpha,mse_data_theta,mse_data_dele, mse_phy
    return mse_data, mse_phy

# loading previously trained weights
model.load_weights("./model_weights/weights")

# custom trainning loop
for epoch in range(epochs):

    # training model
    mse_data,mse_phy = training_function()
    #  mse_data_V,mse_data_alpha,mse_data_theta,mse_data_dele,mse_phy = training_function()

    #  mse_data = np.sum([mse_data_V,mse_data_alpha,mse_data_theta,mse_data_dele])

    #  mse_data = mse_data_V*1.0
    #  mse_data = mse_data_alpha*1.0
    #  mse_data = mse_data_theta*1.0
    #  mse_data = mse_data_dele*1.0
    #  mse_data = mse_data_V + mse_data_alpha + mse_data_theta + mse_data_dele

    # getting coefficients
    CD_0 = model.CD_0.numpy()
    CD_alpha = model.CD_alpha.numpy()
    CD_dele = model.CD_dele.numpy()

    print("epoch : ",epoch+1,
          "\t mse_data : ", np.round(mse_data,8),
          " mse_phy : ", np.round(mse_phy,8))
          #  "\t mse_data : ", np.round(mse_data.numpy(),8),
          #  " mse_phy : ", np.round(mse_phy.numpy(),8))

# saving weights
model.save_weights(os.getcwd()+"/model_weights/weights")

# prediction-------------------------------------------------------------------
V_pred,alpha_pred,theta_pred,dele_pred,CD_pred = model.predict_x([time])

V_res,coeff_res = model.predict_f([time])

V_res = abs(V_res)
coeff_res = abs(coeff_res)

# scaling up CD value
fid_MM = pd.read_csv("../../02_dataProcessing/minMax_computed_data.csv")
CD_min = fid_MM['CD'].iloc[2]
CD_max = fid_MM['CD'].iloc[3]
CD_pred_s = CD_pred*(CD_max-CD_min) + CD_min
CD_s = CD*(CD_max-CD_min) + CD_min

# error computation
CD_error = np.abs(CD_pred - CD)/np.max(CD)*100.0

time = time.flatten()

plt.rcParams.update({'font.size':15})
#  plt.figure(figsize=(16,8))
plt.figure()
#  plt.plot(CD_pred_s,'-b', label = "predicted")
#  plt.plot(CD_s,'-r', label = "actual")
plt.plot(CD_pred,'-b', label = "predicted")
plt.plot(CD,'-r', label = "actual")
plt.grid()
plt.xlabel("data count")
plt.ylabel("CD")
plt.legend()
plt.title("CD")
plt.savefig("CD.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(time,V_pred,'-b',label = "pred")
plt.plot(fid_nd_V["time"],V,'*r',label = "actual")
plt.grid()
plt.legend()
plt.title("V")
plt.xlabel("data count")
plt.ylabel("V")
plt.savefig("V.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(time,alpha_pred,'-b',label = "pred")
plt.plot(fid_nd_alpha["time"],alpha,'*r',label = "actual")
plt.grid()
plt.legend()
plt.title(r"$\alpha$ prediction")
plt.xlabel("data count")
plt.ylabel(r"$\alpha$")
plt.savefig("alpha.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(time,theta_pred,'-b',label = "pred")
plt.plot(fid_nd_theta["time"],theta,'*r',label = "actual")
plt.grid()
plt.legend()
plt.title(r"$\theta$ prediction")
plt.xlabel("data count")
plt.ylabel(r"$\theta$")
plt.savefig("theta.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(time,dele_pred,'-b',label = "pred")
plt.plot(fid_nd_dele["time"],dele,'*r',label = "actual")
plt.grid()
plt.legend()
plt.title(r"$\delta_e$ prediction")
plt.xlabel("data count")
plt.ylabel(r"$\delta_e$")
plt.savefig("dele.png", dpi = 150, bbox_inches = "tight")

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
print("CD_0     = ", model.CD_0.numpy())
print("CD_alpha = ", model.CD_alpha.numpy())
print("CD_dele  = ", model.CD_dele.numpy())

# writing predicted and actual values to the file
fid = pd.DataFrame(np.transpose([CD_s.flatten(),CD_pred_s.numpy().flatten()]),
                   columns = ["CD_act","CD_pred"])
fid['Error'] = abs(fid['CD_act'] - fid['CD_pred'])/fid['CD_act']*100.0
fid.to_csv(os.getcwd()+"/CD_predicted_data.csv", index = None)
del fid

fid = pd.DataFrame(np.transpose([time,V_pred.numpy().flatten(),
                                 alpha_pred.numpy().flatten(),
                                 theta_pred.numpy().flatten(),
                                 dele_pred.numpy().flatten()]),
                   columns = ["time","V","alpha","theta","dele"])
fid.to_csv(os.getcwd()+"/SAC_predicted_data.csv",index=None)

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
