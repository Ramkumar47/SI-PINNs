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
        NeuronsCount = 15

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
        self.min_q     = K.constant(-0.234929)
        self.min_Cm    = K.constant(-0.045476)
        self.min_time  = K.constant(-1.478)

        self.max_q     = K.constant(0.252898)
        self.max_Cm    = K.constant(0.077452)
        self.max_time  = K.constant(16.258)

        # defining concat layer
        self.concat = tf.keras.layers.Concatenate(axis = -1, name = 'concat')

        # defining dense layer 1
        self.D1 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d1")

        # defining dense layer 2
        self.D2 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d2")

        # defining dense layer 3
        self.D3 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d3")

        # defining dense layer 4
        self.D4 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d4")

        # defining dense layer 5
        self.D5 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d5")

        # defining dense layer 6
        self.D6 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d6")

        # defining dense layer 7
        self.D7 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d7")

        # defining dense layer 8
        self.D8 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d8")

        # defining dense layer 9
        self.D9 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d9")

        # defining dense layer 10
        self.D10 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d10")

        # defining dense layer 11
        self.D11 = tf.keras.layers.Dense(units = NeuronsCount, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "d11")

        # defining output layer :
        self.Cm_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "cm")

        # defining output layer :
        self.q_layer = tf.keras.layers.Dense(units = 1, activation = activation,
                                      kernel_initializer = kern_init,
                                      bias_initializer = bias_init,
                                      kernel_regularizer = regularizer,
                                      bias_regularizer = regularizer,
                                      name = "q")

        # switching traibability of layers

        # Cm network
        self.D1.trainable       = True
        self.D2.trainable       = True
        self.D3.trainable       = True
        self.D4.trainable       = True
        self.D10.trainable      = True
        self.Cm_layer.trainable = True

        # q network
        self.D5.trainable       = False
        self.D6.trainable       = False
        self.D7.trainable       = False
        self.D8.trainable       = False
        self.D9.trainable       = False
        self.q_layer.trainable  = False


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
        d_out1 = self.D1(time)
        d_out1 = self.D2(d_out1)
        d_out1 = self.D3(d_out1)
        d_out1 = self.D4(d_out1)
        Cm     = self.Cm_layer(d_out1)

        d_out2 = self.D5(time)
        d_out2 = self.D6(d_out2)
        d_out2 = self.D7(d_out2)
        d_out2 = self.D8(d_out2)
        d_out2 = self.D9(d_out2)
        q     = self.q_layer(d_out2)

        return q,Cm

    # defining PINN function
    def PINNFunction(self, time,V,alpha,dele):

        # computing NN output
        q,Cm = self.NeuralFunction(time)

        # computing the derivative
        dqdt = K.gradients(q,time)[0]*(self.max_q-self.min_q)/(self.max_time-self.min_time)

        # scaling up the values
        q     = q*(self.max_q - self.min_q) + self.min_q
        Cm    = Cm*(self.max_Cm - self.min_Cm) + self.min_Cm

        # computing dynamic pressure
        q_bar = 0.5*self.rho*V**2

        # framing equations and computing residual
        res_q     = q_bar*self.S*self.c_bar/self.Iyy*Cm - dqdt

        # estimating the coefficient values
        res_coeff = self.Cm_0 + self.Cm_alpha*alpha + self.Cm_q*q*self.c_bar/2.0/V + self.Cm_dele*dele - Cm

        return res_q, res_coeff

    @tf.function
    def predict_x(self, inputs):

        # inputs
        time = inputs[0]

        # computing NN output
        q,Cm = self.NeuralFunction(time)

        # returning outputs
        return q,Cm

    @tf.function
    def predict_f(self, inputs):

        # inputs
        time  = inputs[0]
        V     = inputs[1]
        alpha = inputs[2]
        dele  = inputs[3]

        # computing NN output
        res = self.PINNFunction(time,V,alpha,dele)

        # returning outputs
        return res

    def call(self, inputs, training = True):

        """
            made dummy
        """
        return 0
