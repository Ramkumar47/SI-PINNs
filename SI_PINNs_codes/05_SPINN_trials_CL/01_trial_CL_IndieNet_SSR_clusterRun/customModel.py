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
        kern_init          = kernel_initializer
        bias_init          = bias_initializer
        activation         = self.tanh
        NeuronsCount_alpha = 15
        NeuronsCount_CL    = 10

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
        self.CL_0     = K.variable(0.365)
        self.CL_alpha = K.variable(4.972)
        self.CL_q     = K.variable(37.3)
        self.CL_dele  = K.variable(0.26)
        self.Cm_0     = K.variable(0.05)
        self.Cm_alpha = K.variable(-0.48)
        self.Cm_q     = K.variable(-11.3)
        self.Cm_dele  = K.variable(-1.008)
        self.CD_0     = K.variable(0.5, constraint=tf.keras.constraints.non_neg())
        self.CD_alpha = K.variable(0.5, constraint=tf.keras.constraints.non_neg())
        self.CD_dele  = K.variable(0.5, constraint=tf.keras.constraints.non_neg())

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
        self.min_alpha = K.constant(-0.016335)
        self.min_CL    = K.constant(0.212871)
        self.min_time  = K.constant(-1.478)

        self.max_alpha = K.constant(0.171416)
        self.max_CL    = K.constant(1.285295)
        self.max_time  = K.constant(16.258)

        # defining concat layer
        self.concat = tf.keras.layers.Concatenate(axis = -1, name = 'concat')

        # defining layers for alpha
        # defining dense layer 1
        self.L1a = tf.keras.layers.Dense(units = NeuronsCount_alpha, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l1a")

        # defining dense layer 2
        self.L2a = tf.keras.layers.Dense(units = NeuronsCount_alpha, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l2a")

        # defining dense layer 3
        self.L3a = tf.keras.layers.Dense(units = NeuronsCount_alpha, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l3a")

        # defining dense layer 4
        self.L4a = tf.keras.layers.Dense(units = NeuronsCount_alpha, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l4a")

        # defining dense layer 5
        self.L5a = tf.keras.layers.Dense(units = NeuronsCount_alpha, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l5a")

        # defining layers for CL
        # defining dense layer 6
        self.L1CL = tf.keras.layers.Dense(units = NeuronsCount_CL, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l1cl")

        # defining dense layer 7
        self.L2CL = tf.keras.layers.Dense(units = NeuronsCount_CL, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l2cl")

        # defining dense layer 8
        self.L3CL = tf.keras.layers.Dense(units = NeuronsCount_CL, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l3cl")

        # defining dense layer 9
        self.L4CL = tf.keras.layers.Dense(units = NeuronsCount_CL, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l4cl")

        # defining dense layer 10
        self.L5CL = tf.keras.layers.Dense(units = NeuronsCount_CL, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "l5cl")

        # defining output layer :
        self.CL_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "cl")

        # defining output layer :
        self.alpha_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "alpha")

        # switching traibability of layers

        # CL network
        self.L1CL.trainable     = True
        self.L2CL.trainable     = True
        self.L3CL.trainable     = True
        self.L4CL.trainable     = True
        self.L5CL.trainable     = True
        self.CL_layer.trainable = True

        # alpha network
        self.L1a.trainable         = False
        self.L2a.trainable         = False
        self.L3a.trainable         = False
        self.L4a.trainable         = False
        self.L5a.trainable         = False
        self.alpha_layer.trainable = False


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
        d_out1 = self.L1CL(time)
        d_out1 = self.L2CL(d_out1)
        d_out1 = self.L3CL(d_out1)
        d_out1 = self.L4CL(d_out1)
        CL     = self.CL_layer(d_out1)

        d_out2 = self.L1a(time)
        d_out2 = self.L2a(d_out2)
        d_out2 = self.L3a(d_out2)
        d_out2 = self.L4a(d_out2)
        d_out2 = self.L5a(d_out2)
        alpha  = self.alpha_layer(d_out2)

        return alpha,CL

    # defining PINN function
    def PINNFunction(self, time,V,q,theta,dele):

        # computing NN output
        alpha,CL = self.NeuralFunction(time)

        # computing the derivative
        dalphadt = K.gradients(alpha,time)[0]*(self.max_alpha-self.min_alpha)/(self.max_time-self.min_time)

        # scaling up the values
        alpha     = alpha*(self.max_alpha - self.min_alpha) + self.min_alpha
        CL    = CL*(self.max_CL - self.min_CL) + self.min_CL

        # computing dynamic pressure
        q_bar = 0.5*self.rho*V**2

        # framing equations and computing residual
        res_alpha     = -q_bar*self.S/self.m/V*CL + q - self.T/self.m/V*K.sin(alpha) + self.g/V*K.cos(alpha-theta) - dalphadt

        # estimating the coefficient values
        res_coeff = self.CL_0 + self.CL_alpha*alpha + self.CL_q*q*self.c_bar/2.0/V + self.CL_dele*dele - CL

        return res_alpha, res_coeff

    @tf.function
    def predict_x(self, inputs):

        # inputs
        time = inputs[0]

        # computing NN output
        alpha,CL = self.NeuralFunction(time)

        # returning outputs
        return alpha,CL

    @tf.function
    def predict_f(self, inputs):

        # inputs
        time  = inputs[0]
        V     = inputs[1]
        q     = inputs[2]
        theta = inputs[3]
        dele  = inputs[4]

        # computing NN output
        res = self.PINNFunction(time,V,q,theta,dele)

        # returning outputs
        return res

    def call(self, inputs, training = True):

        """
            made dummy
        """
        return 0
