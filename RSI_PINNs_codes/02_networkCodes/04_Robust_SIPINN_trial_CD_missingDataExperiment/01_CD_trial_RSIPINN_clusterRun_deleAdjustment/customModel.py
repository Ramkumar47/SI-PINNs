#!/bin/python3
"""----------------------------------------------------------------------------
Custom Tensorflow Model build for Robust SI-PINN (RSI-PINN)
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

# hiding unnecessary warnings--------------------------------------------------
# tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# setting float type
tf.keras.backend.set_floatx('float64')

# disabling eager execution
#  tf.compat.v1.disable_eager_execution()
#  tf.compat.v1.enable_eager_execution()

tf.compat.v1.experimental.output_all_intermediates(True)

# building model---------------------------------------------------------------

class CustomModel(tf.keras.Model): #  with unscaled inputs
    def __init__(self, kernel_initializer, bias_initializer,
            activation = 'selu', regularizer = None):
        super(CustomModel, self).__init__(name = "custom_model")

        # defining layer parameters
        kern_init             = kernel_initializer
        bias_init             = bias_initializer
        activation            = self.tanh
        NeuronsCount_Vnet     = 15
        NeuronsCount_alphanet = 15
        NeuronsCount_thetanet = 15
        NeuronsCount_delenet  = 15
        NeuronsCount_CDnet    = 15

        # network enabling flags
        V_net_flag     = False
        alpha_net_flag = False
        theta_net_flag = False
        dele_net_flag  = False
        CD_net_flag    = True

        # defining constant parameters
        self.c_bar  = K.constant(1.211)  # mean a/d chord
        self.b      = K.constant(10.47)  # wing span
        self.AR     = K.constant(8.8)    # aspect ratio
        self.S      = K.constant(12.47)  # wing area
        self.m      = K.constant(750)    # mass
        self.Iyy    = K.constant(907.0)  # mmt of inertia
        self.T      = K.constant(1136.0) # thrust
        self.g      = K.constant(9.81)   # acc. due to gravity
        self.rho    = K.constant(1.225)  # density
        self.deltaT = K.constant(0.02)   # timestep

        # defining trainable variables for a/d derivatives
        self.CL_0     = K.constant(0.5)
        self.CL_alpha = K.constant(0.5)
        self.CL_q     = K.constant(0.5)
        self.CL_dele  = K.constant(0.5)
        self.Cm_0     = K.constant(0.5)
        self.Cm_alpha = K.constant(0.5)
        self.Cm_q     = K.constant(0.5)
        self.Cm_dele  = K.constant(0.5)
        self.CD_0     = K.variable(0.5)
        self.CD_alpha = K.variable(0.5)
        self.CD_dele  = K.variable(0.5)

        #  # defining trainable variables for a/d derivatives
        #  self.CL_0     = K.constant(0.365)
        #  self.CL_alpha = K.constant(4.972)
        #  self.CL_q     = K.constant(37.3)
        #  self.CL_dele  = K.constant(0.26)
        #  self.Cm_0     = K.constant(0.05)
        #  self.Cm_alpha = K.constant(-0.48)
        #  self.Cm_q     = K.constant(-11.3)
        #  self.Cm_dele  = K.constant(-1.008)
        #  self.CD_0     = K.constant(0.036)
        #  self.CD_alpha = K.constant(0.041)
        #  self.CD_dele  = K.constant(0.026)

        # hardcoding minmax values of state and control variables
        self.min_V     = K.constant(30.15026085851284)
        self.min_CD    = K.constant(0.0367230810715945)
        self.min_time  = K.constant(-0.8211111111111112)
        self.min_alpha = K.constant(-0.0093811815732616)
        self.min_q     = K.constant(-0.2168608828435358)
        self.min_dele  = K.constant(-0.0525219)
        self.min_theta = K.constant(-0.0275297103209113)

        self.max_V     = K.constant(42.0139810074467)
        self.max_CD    = K.constant(0.0422272638086755)
        self.max_time  = K.constant(15.601111111111113)
        self.max_alpha = K.constant(0.1644624902763348)
        self.max_q     = K.constant(0.234830347351212)
        self.max_dele  = K.constant(0.0919401)
        self.max_theta = K.constant(0.5876977789584745)

        # defining concat layer
        self.concat = tf.keras.layers.Concatenate(axis = -1, name = 'concat')

        # defining layers for v_net
        # defining dense layer 1
        self.L1V = tf.keras.layers.Dense(units = NeuronsCount_Vnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l1v")

        # defining dense layer 2
        self.L2V = tf.keras.layers.Dense(units = NeuronsCount_Vnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l2v")

        # defining dense layer 3
        self.L3V = tf.keras.layers.Dense(units = NeuronsCount_Vnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l3v")

        # defining dense layer 4
        self.L4V = tf.keras.layers.Dense(units = NeuronsCount_Vnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l4v")

        # defining dense layer 5
        self.L5V = tf.keras.layers.Dense(units = NeuronsCount_Vnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l5v")

        # defining layers for alpha_net
        # defining dense layer 1
        self.L1alpha = tf.keras.layers.Dense(units = NeuronsCount_alphanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l1alpha")

        # defining dense layer 2
        self.L2alpha = tf.keras.layers.Dense(units = NeuronsCount_alphanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l2alpha")

        # defining dense layer 3
        self.L3alpha = tf.keras.layers.Dense(units = NeuronsCount_alphanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l3alpha")

        # defining dense layer 4
        self.L4alpha = tf.keras.layers.Dense(units = NeuronsCount_alphanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l4alpha")

        # defining dense layer 5
        self.L5alpha = tf.keras.layers.Dense(units = NeuronsCount_alphanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l5alpha")

        # defining layers for theta_net
        # defining dense layer 1
        self.L1theta = tf.keras.layers.Dense(units = NeuronsCount_thetanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l1theta")

        # defining dense layer 2
        self.L2theta = tf.keras.layers.Dense(units = NeuronsCount_thetanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l2theta")

        # defining dense layer 3
        self.L3theta = tf.keras.layers.Dense(units = NeuronsCount_thetanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l3theta")

        # defining dense layer 4
        self.L4theta = tf.keras.layers.Dense(units = NeuronsCount_thetanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l4theta")

        # defining dense layer 5
        self.L5theta = tf.keras.layers.Dense(units = NeuronsCount_thetanet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l5theta")

        # defining layers for dele_net
        # defining dense layer 1
        self.L1dele = tf.keras.layers.Dense(units = NeuronsCount_delenet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l1dele")

        # defining dense layer 2
        self.L2dele = tf.keras.layers.Dense(units = NeuronsCount_delenet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l2dele")

        # defining dense layer 3
        self.L3dele = tf.keras.layers.Dense(units = NeuronsCount_delenet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l3dele")

        # defining dense layer 4
        self.L4dele = tf.keras.layers.Dense(units = NeuronsCount_delenet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l4dele")

        # defining dense layer 5
        self.L5dele = tf.keras.layers.Dense(units = NeuronsCount_delenet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l5dele")
        # defining dense layers for CD
        # defining Dense layer 1
        self.L1CD = tf.keras.layers.Dense(units = NeuronsCount_CDnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l1cd")

        # defining dense layer 2
        self.L2CD = tf.keras.layers.Dense(units = NeuronsCount_CDnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l2cd")

        # defining dense layer 3
        self.L3CD = tf.keras.layers.Dense(units = NeuronsCount_CDnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l3cd")

        # defining dense layer 4
        self.L4CD = tf.keras.layers.Dense(units = NeuronsCount_CDnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l4cd")

        # defining dense layer 5
        self.L5CD = tf.keras.layers.Dense(units = NeuronsCount_CDnet, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l5cd")

        # defining output layers :
        self.CD_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "cd")
        self.V_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "v")
        self.alpha_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "alpha")
        self.theta_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "theta")
        self.dele_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "dele")

        # switching traibability of layers

        # CD network
        self.L1CD.trainable     = CD_net_flag
        self.L2CD.trainable     = CD_net_flag
        self.L3CD.trainable     = CD_net_flag
        self.L4CD.trainable     = CD_net_flag
        self.L5CD.trainable     = CD_net_flag
        self.CD_layer.trainable = CD_net_flag

        # V network
        self.L1V.trainable      = V_net_flag
        self.L2V.trainable      = V_net_flag
        self.L3V.trainable      = V_net_flag
        self.L4V.trainable      = V_net_flag
        self.L5V.trainable      = V_net_flag
        self.V_layer.trainable  = V_net_flag

        # alpha network
        self.L1alpha.trainable     = alpha_net_flag
        self.L2alpha.trainable     = alpha_net_flag
        self.L3alpha.trainable     = alpha_net_flag
        self.L4alpha.trainable     = alpha_net_flag
        self.L5alpha.trainable     = alpha_net_flag
        self.alpha_layer.trainable = alpha_net_flag

        # theta network
        self.L1theta.trainable     = theta_net_flag
        self.L2theta.trainable     = theta_net_flag
        self.L3theta.trainable     = theta_net_flag
        self.L4theta.trainable     = theta_net_flag
        self.L5theta.trainable     = theta_net_flag
        self.theta_layer.trainable = theta_net_flag

        # dele network
        self.L1dele.trainable     = dele_net_flag
        self.L2dele.trainable     = dele_net_flag
        self.L3dele.trainable     = dele_net_flag
        self.L4dele.trainable     = dele_net_flag
        self.L5dele.trainable     = dele_net_flag
        self.dele_layer.trainable = dele_net_flag

    def tanh(self,x):
        val = tf.keras.activations.tanh(x)

        return val

    # defining neural function for RKNN
    @tf.function
    def NeuralFunction(self,time):

        # training V network
        d_out1 = self.L1V(time)
        d_out1 = self.L2V(d_out1)
        d_out1 = self.L3V(d_out1)
        d_out1 = self.L4V(d_out1)
        d_out1 = self.L5V(d_out1)
        V      = self.V_layer(d_out1)

        # training alpha network
        d_out2 = self.L1alpha(time)
        d_out2 = self.L2alpha(d_out2)
        d_out2 = self.L3alpha(d_out2)
        d_out2 = self.L4alpha(d_out2)
        d_out2 = self.L5alpha(d_out2)
        alpha  = self.alpha_layer(d_out2)

        # training theta network
        d_out3 = self.L1theta(time)
        d_out3 = self.L2theta(d_out3)
        d_out3 = self.L3theta(d_out3)
        d_out3 = self.L4theta(d_out3)
        d_out3 = self.L5theta(d_out3)
        theta  = self.theta_layer(d_out3)

        # training dele network
        d_out4 = self.L1dele(time)
        d_out4 = self.L2dele(d_out4)
        d_out4 = self.L3dele(d_out4)
        d_out4 = self.L4dele(d_out4)
        d_out4 = self.L5dele(d_out4)
        dele  = self.dele_layer(d_out4)

        # training CD network
        d_out5 = self.concat([alpha,dele])
        d_out5 = self.L1CD(d_out5)
        d_out5 = self.L2CD(d_out5)
        d_out5 = self.L3CD(d_out5)
        d_out5 = self.L4CD(d_out5)
        d_out5 = self.L5CD(d_out5)
        CD     = self.CD_layer(d_out5)

        return V,alpha,theta,dele,CD

    # defining PINN function
    def PINNFunction(self, time):

        # computing NN output
        V,alpha,theta,dele,CD = self.NeuralFunction(time)

        # computing the derivative
        dVdt = K.gradients(V,time)[0]*(self.max_V-self.min_V)/(self.max_time-self.min_time)

        # scaling up the values
        V     = V*(self.max_V - self.min_V) + self.min_V
        CD    = CD*(self.max_CD - self.min_CD) + self.min_CD
        alpha = alpha*(self.max_alpha - self.min_alpha) + self.min_alpha
        theta = theta*(self.max_theta - self.min_theta) + self.min_theta
        dele  = dele*(self.max_dele - self.min_dele) + self.min_dele

        # computing dynamic pressure
        q_bar = 0.5*self.rho*V**2

        # framing equations and computing residual
        res_V     = -q_bar*self.S/self.m*CD + self.T/self.m*K.cos(alpha) + self.g*K.sin(alpha - theta) - dVdt

        # estimating the coefficient values
        res_coeff = self.CD_0 + self.CD_alpha*alpha + self.CD_dele*dele - CD

        return res_V, res_coeff

    @tf.function
    def predict_x(self, inputs):

        # inputs
        time  = inputs[0]

        # computing NN output
        V,alpha,theta,dele,CD = self.NeuralFunction(time)

        # returning outputs
        return V,alpha,theta,dele,CD

    @tf.function
    def predict_f(self, inputs):

        # inputs
        time  = inputs[0]

        # computing NN output
        res = self.PINNFunction(time)

        # returning outputs
        return res

    def call(self, inputs, training = True):

        """
            made dummy
        """
        return 0
