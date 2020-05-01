# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:09:57 2019

@author: AMSabry
"""
import tensorflow as tf


def scSE(input_layer, reduce_to_units = None):
    ## Channel wise Squeeze and Excitation:
    shapes = input_layer.get_shape().as_list()
    avg_pooled = tf.nn.avg_pool(input_layer, 
                                [1, shapes[1], shapes[2], 1], 
                                [1, 1, 1, 1], 
                                padding = 'VALID')
    if reduce_to_units == None:
        FC1 = tf.contrib.layers.fully_connected(avg_pooled, shapes[-1], activation_fn = tf.nn.relu)
    else: FC1 = tf.contrib.layers.fully_connected(avg_pooled, reduce_to_units, activation_fn = tf.nn.relu)
    FC2 = tf.contrib.layers.fully_connected(FC1, shapes[-1], activation_fn = tf.nn.sigmoid)
    final = input_layer * FC2
    
    ## Spacial wise Squeeze and Excitation:
    weights = tf.Variable(tf.truncated_normal([1, 1, shapes[-1], 1]), dtype = tf.float64)
    bias = tf.Variable(tf.zeros([1]))
    conved_excitor = tf.nn.conv2d(input_layer, weights, [1, 1, 1, 1], 'VALID')
    conved_excitor = tf.nn.relu(conved_excitor+bias)
    final_ = input_layer * conved_excitor
    scse = final+final_
    return scse

