#!/bin/python3
"""----------------------------------------------------------------------------
Custom Tensorflow Model build for SI-PINN
Determinant SI-PINN with RK4 method
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
        kern_init    = kernel_initializer
        bias_init    = bias_initializer
        activation   = self.tanh
        NeuronsCount_Vnet = 15
        NeuronsCount_CDnet = 10

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
        self.CL_0     = K.variable(0.365, constraint=tf.keras.constraints.non_neg())
        self.CL_alpha = K.variable(4.972, constraint=tf.keras.constraints.non_neg())
        self.CL_q     = K.variable(37.3, constraint=tf.keras.constraints.non_neg())
        self.CL_dele  = K.variable(0.26, constraint=tf.keras.constraints.non_neg())
        self.Cm_0     = K.variable(0.05)
        self.Cm_alpha = K.variable(-0.48)
        self.Cm_q     = K.variable(-11.3)
        self.Cm_dele  = K.variable(-1.008)
        #  self.CD_0     = K.variable(0.5, constraint=tf.keras.constraints.non_neg())
        #  self.CD_alpha = K.variable(0.5, constraint=tf.keras.constraints.non_neg())
        #  self.CD_dele  = K.variable(0.5, constraint=tf.keras.constraints.non_neg())
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
        self.min_V     = K.constant(29.675712)
        self.min_CD    = K.constant(0.036503)
        self.min_time  = K.constant(-1.478)

        self.max_V     = K.constant(42.488530)
        self.max_CD    = K.constant(0.042447)
        self.max_time  = K.constant(16.258)

        # defining concat layer
        self.concat = tf.keras.layers.Concatenate(axis = -1, name = 'concat')

        # defining layers for V_net
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

        # defining dense layers for CD
        # defining dense layer 1
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

        # defining output layer :
        self.CD_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "cd")

        # defining output layer :
        self.V_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "v")

        # switching traibability of layers

        # CD network
        self.L1CD.trainable     = True
        self.L2CD.trainable     = True
        self.L3CD.trainable     = True
        self.L4CD.trainable     = True
        self.L5CD.trainable     = True
        self.CD_layer.trainable = True

        # V network
        self.L1V.trainable      = False
        self.L2V.trainable      = False
        self.L3V.trainable      = False
        self.L4V.trainable      = False
        self.L5V.trainable      = False
        self.V_layer.trainable  = False


    def tanh(self,x):
        val = tf.keras.activations.tanh(x)

        return val

    def softplus(self,x):
        # softplus activation function
        val = K.log(1.0 + K.exp(x))
        return val

    # defining neural function for RKNN
    @tf.function
    def NeuralFunction(self,time):

        # passing input through dense layers
        d_out1 = self.L1CD(time)
        d_out1 = self.L2CD(d_out1)
        d_out1 = self.L3CD(d_out1)
        d_out1 = self.L4CD(d_out1)
        d_out1 = self.L5CD(d_out1)
        CD     = self.CD_layer(d_out1)

        d_out2 = self.L1V(time)
        d_out2 = self.L2V(d_out2)
        d_out2 = self.L3V(d_out2)
        d_out2 = self.L4V(d_out2)
        d_out2 = self.L5V(d_out2)
        V      = self.V_layer(d_out2)

        return V,CD

    # defining PINN function
    def PINNFunction(self, time,theta,alpha,dele):

        # computing NN output
        V,CD = self.NeuralFunction(time)

        # computing the derivative
        dVdt = K.gradients(V,time)[0]*(self.max_V-self.min_V)/(self.max_time-self.min_time)

        # scaling up the values
        V     = V*(self.max_V - self.min_V) + self.min_V
        CD    = CD*(self.max_CD - self.min_CD) + self.min_CD

        # computing dynamic pressure
        q_bar = 0.5*self.rho*V**2

        # framing equations and computing residual
        res_V     = -q_bar*self.S/self.m*CD + self.T/self.m*K.cos(alpha) + self.g*K.sin(alpha - theta) - dVdt
        #  res_V     = -q_bar*self.S/self.m + self.T/self.m*K.cos(alpha)/CD + self.g*K.sin(alpha - theta)/CD - dVdt/CD

        # estimating the coefficient values
        res_coeff = self.CD_0 + self.CD_alpha*alpha + self.CD_dele*dele - CD

        return res_V, res_coeff

    @tf.function
    def predict_x(self, inputs):

        # inputs
        time = inputs[0]

        # computing NN output
        V,CD = self.NeuralFunction(time)

        # returning outputs
        return V,CD

    @tf.function
    def predict_f(self, inputs):

        # inputs
        time  = inputs[0]
        theta = inputs[1]
        alpha = inputs[2]
        dele  = inputs[3]

        # computing NN output
        res = self.PINNFunction(time,theta,alpha,dele)

        # returning outputs
        return res

    def call(self, inputs, training = True):

        """
            made dummy
        """
        return 0
