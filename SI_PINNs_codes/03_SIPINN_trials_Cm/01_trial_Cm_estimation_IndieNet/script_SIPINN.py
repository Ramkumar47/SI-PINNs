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
epochs = 800
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
fid_cd = pd.read_csv("../../01_numericalSolution/computed_data.csv")
fid_nd = pd.read_csv("../../02_dataProcessing/normalized_data.csv")

# getting computed fields to be used in PI function
V = fid_cd['V'].to_numpy()[:,None]
alpha = fid_cd['alpha'].to_numpy()[:,None]
dele = fid_cd['dele'].to_numpy()[:,None]

# getting normalized data to be fed in the model for coeff. estimation
q = fid_nd['q'].to_numpy()[:,None]
time = fid_nd['time'].to_numpy()[:,None]

# getting ad coefficients for output comparison
Cm = fid_nd['Cm'].to_numpy()[:,None]


# training model---------------------------------------------------------------

# defining loss function
lossFunc = tf.keras.losses.MeanSquaredError()

# training function
@tf.function
def training_function():

    # evaluating model for data output
    q_pred, Cm_pred = model.predict_x([time])
    mse_data        = lossFunc(q_pred, q)

    # evaluating model for PINN CD output
    res_q,res_coeff = model.predict_f([time,V,alpha,dele])
    res_q_loss      = lossFunc(res_q, res_q*K.constant(0.0))
    res_coeff_loss  = lossFunc(res_coeff, res_coeff*K.constant(0.0))
    mse_phy         = res_q_loss + res_coeff_loss

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

#  # custom trainning loop
#  for epoch in range(epochs):
#
#      # training model
#      mse_data, mse_phy = training_function()
#
#      # getting coefficients
#      Cm_0 = model.Cm_0.numpy()
#      Cm_alpha = model.Cm_alpha.numpy()
#      Cm_q = model.Cm_q.numpy()
#      Cm_dele = model.Cm_dele.numpy()
#
#      print("epoch : ",epoch+1,
#            "\t mse_data : ", np.round(mse_data.numpy(),8),
#            " mse_phy : ", np.round(mse_phy.numpy(),8))
#
#      #  print("epoch : ",epoch+1,
#      #        "\t mse_data : ", np.round(mse_data.numpy(),8),
#      #        " mse_phy : ", np.round(mse_phy.numpy(),8),
#      #        "\t Cm_0 : ", Cm_0,
#      #        "; Cm_alpha : ", Cm_alpha,
#      #        "; Cm_q : ", Cm_q,
#      #        "; Cm_dele : ", Cm_dele)
#
#  # saving weights
#  model.save_weights(os.getcwd()+"/model_weights/weights")

# prediction-------------------------------------------------------------------
q_pred,Cm_pred = model.predict_x([time])

q_res,coeff_res = model.predict_f([time,V,alpha,dele])

q_res = abs(q_res)
coeff_res = abs(coeff_res)

# scaling up CD value
fid_MM = pd.read_csv("../../02_dataProcessing/minMaxValues.csv")
Cm_min = fid_MM['Cm'].iloc[2]
Cm_max = fid_MM['Cm'].iloc[3]
Cm_pred_s = Cm_pred*(Cm_max-Cm_min) + Cm_min
Cm_s = Cm*(Cm_max-Cm_min) + Cm_min

# error computation
Cm_error = np.abs(Cm_pred - Cm)/np.max(Cm)*100.0

#  plt.figure(figsize=(16,8))
#  plt.subplot(1,2,1)
#  plt.plot(Cm_pred_s,'-b', label = "predicted")
#  plt.plot(Cm_s,'-r', label = "actual")
#  plt.grid()
#  plt.xlabel("data count")
#  plt.ylabel("Cm")
#  plt.legend()
#  plt.title("Cm")
#  plt.subplot(1,2,2)
#  plt.plot(Cm_error,'-b')
#  plt.xlabel("data count")
#  plt.ylabel("error percentage")
#  plt.yscale("log")
#  plt.title("Cm error percentage")
#  plt.grid()
#  plt.savefig("Cm.png", dpi = 150)
#
#  plt.figure()
#  plt.plot(q_pred,'-b',label = "pred")
#  plt.plot(q,'-r',label = "actual")
#  plt.grid()
#  plt.legend()
#  plt.title("q")
#  plt.savefig("q.png", dpi = 150)
#
#  plt.figure()
#  plt.plot(q_res,'-b',label = "q")
#  plt.plot(coeff_res,'-r',label = "coeff. eqn")
#  plt.grid()
#  plt.yscale('log')
#  plt.legend()
#  plt.title("residual")
#  plt.savefig("residual.png", dpi = 150)

plt.rcParams.update({'font.size':15})
#  plt.figure(figsize=(16,8))
plt.figure()
plt.plot(Cm_pred_s,'-b', label = "predicted")
plt.plot(Cm_s,'-r', label = "actual")
plt.grid()
plt.xlabel("data count")
plt.ylabel(r"$C_m$")
plt.legend()
plt.title(r"$C_m$")
plt.savefig("Cm.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(q_pred,'-b',label = "pred")
plt.plot(q,'-r',label = "actual")
plt.grid()
plt.legend()
plt.title(r"$q$")
plt.xlabel("data count")
plt.ylabel(r"$q$")
plt.savefig("q.png", dpi = 150, bbox_inches = "tight")

plt.figure()
plt.plot(q_res,'-b',label = r"$q$")
plt.plot(coeff_res,'-r',label = r"$C_m$")
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
print("Cm Error :")
print("\t max = ",Cm_error.max())
print("\t min = ",Cm_error.min())
print("\t avg = ",Cm_error.mean())


# printing model trainable parameters
print("\nModel computed A/D coefficients : ")
#  print("CL_0     = ", model.CL_0.numpy())
#  print("CL_alpha = ", model.CL_alpha.numpy())
#  print("CL_q     = ", model.CL_q.numpy())
#  print("CL_dele  = ", model.CL_dele.numpy())
#  print("CD_0     = ", model.CD_0.numpy())
#  print("CD_alpha = ", model.CD_alpha.numpy())
#  print("CD_dele  = ", model.CD_dele.numpy())
print("Cm_0     = ", model.Cm_0.numpy())
print("Cm_alpha = ", model.Cm_alpha.numpy())
print("Cm_q     = ", model.Cm_q.numpy())
print("Cm_dele  = ", model.Cm_dele.numpy())

# writing predicted and actual values to the file
fid = pd.DataFrame(np.transpose([Cm_s.flatten(),Cm_pred_s.numpy().flatten()]),
                   columns = ["Cm_act","Cm_pred"])
fid['Error'] = abs(fid['Cm_act'] - fid['Cm_pred'])/fid['Cm_act']*100.0
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
