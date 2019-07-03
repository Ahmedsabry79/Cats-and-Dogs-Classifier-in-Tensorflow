import tensorflow as tf
#import numpy as np
#import pandas as pd


def conv_2s(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 2, 2, 1], padding = 'SAME')

def conv_1s(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def Maxpool_2s_33(input_layer):
    return tf.nn.max_pool(input_layer, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    

def Inception_Layer_1(Data, is_training, filter_size = 7, channels = 3, n_filters = 128,
                      conv_stride = 2, pool_stride = 2, pool_ksize = 3, weights_name = 'Layer_1', biases_name = 'biases_1'):
    ## Layer:
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    WC_1 = {'Layer_1': tf.Variable(initializer([filter_size, filter_size, channels, n_filters]), name = weights_name)}
    biases_L1 = {'biases_1': tf.Variable(tf.zeros([n_filters]), name = biases_name)}

    ## Layer Conv:
    C1 = tf.nn.conv2d(Data, WC_1['Layer_1'], strides = [1, conv_stride, conv_stride, 1], padding = 'SAME')
    C1 = C1 + biases_L1['biases_1']
    C1 = batch_normalization(C1, 0.95, scale = False, is_training = True)
    C1 = tf.nn.relu(C1)
    C1 = tf.nn.max_pool(C1, ksize = [1, pool_ksize, pool_ksize, 1], strides = [1, pool_stride, pool_stride, 1], padding = 'SAME')
    C1 = tf.nn.local_response_normalization(C1, 5)
    return C1


def Inception_Layer_2(input_layer, is_training, input_filters, filter_size = 3,
                      reduce_to_filters = 64, output_filters = 192,
                      conv_stride = 1, pool_stride = 2, pool_ksize = 3, weights_name = 'Layer_2', biases_name = 'biases_2'):
    ##Layer 2
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    WC_2 = {'Layer_2': tf.Variable(initializer([1, 1, input_filters, reduce_to_filters]), name = weights_name+'_red'),
            'Layer_2_33': tf.Variable(initializer([filter_size, filter_size, reduce_to_filters, output_filters]), name = weights_name+'_conv')}
    
    biases_L2 = {'b_1': tf.Variable(tf.zeros([reduce_to_filters]), name = biases_name+'_red'), 
                 'b_2': tf.Variable(tf.zeros([output_filters]), name = biases_name+'_conv')}

    ## Layer 2 Conv:
    C2 = tf.nn.conv2d(input_layer, WC_2['Layer_2'], strides = [1, 1, 1, 1], padding = 'SAME')
    C2 = C2 + biases_L2['b_1']
    C2 = batch_normalization(C2, 0.95, scale = False, is_training = True)
    C2 = tf.nn.relu(C2)

    C2 = tf.nn.conv2d(C2, WC_2['Layer_2_33'], strides = [1, conv_stride, conv_stride, 1], padding = 'SAME')
    C2 = C2+ biases_L2['b_2']
    C2 = batch_normalization(C2, 0.95, scale = False, is_training = True)
    C2 = tf.nn.relu(C2)
    
    C2 = tf.nn.max_pool(C2, ksize = [1, pool_ksize, pool_ksize, 1], strides = [1, pool_stride, pool_stride, 1], padding = 'SAME')
    C2 = tf.nn.local_response_normalization(C2, 5)
    return C2


def Inception_Layer(input_layer, is_training, input_filters, weights_name, biases_name,
                    reduce_to_filters_3 = 96, output_filters_3 = 128, 
                    reduce_to_filters_5 = 16, output_filters_5 = 32, output_filters_1 = 64,
                    output_filters_pool = 32, conv_stride = 1, pool_stride = 1, pool_ksize = 3):

    ##Weights initialization:
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    WC_reduction_3 = {'red': tf.Variable(initializer([1, 1, input_filters, reduce_to_filters_3]), name = weights_name+'_red_3'),
                      '_33': tf.Variable(initializer([3, 3, reduce_to_filters_3, output_filters_3]), name = weights_name+'_conv_3')}
    biases_3 = {'b_1': tf.Variable(tf.zeros([reduce_to_filters_3]), name = biases_name+'_red_3'), 
                 'b_2': tf.Variable(tf.zeros([output_filters_3]), name = biases_name+'_conv_3')}

    WC_reduction_5 = {'red': tf.Variable(initializer([1, 1, input_filters, reduce_to_filters_5]), name = weights_name+'_red_5'),
                        '_33': tf.Variable(initializer([3, 3, reduce_to_filters_5, output_filters_5]), name = weights_name+'_conv_5')}
    biases_5 = {'b_1': tf.Variable(tf.zeros([reduce_to_filters_5]), name = biases_name+'_red_5'), 
                 'b_2': tf.Variable(tf.zeros([output_filters_5]), name = biases_name+'_conv_5')}

    WC_1 = {'_11': tf.Variable(initializer([1, 1, input_filters, output_filters_1]), name = weights_name+'_1')}
    biases_1 = {'b_1': tf.Variable(tf.zeros([output_filters_1]), name = biases_name+'_1')}

    WC_pool = {'_11': tf.Variable(initializer([1, 1, input_filters, output_filters_pool]), name = weights_name+'_pool')}
    biases_pool = {'b_1': tf.Variable(tf.zeros([output_filters_pool]), name = biases_name+'_pool')}

    ## Layer Conv:
    C3 = conv_1s(input_layer, WC_reduction_3['red'])
    C3 = C3 + biases_3['b_1']
    C3 = batch_normalization(C3, 0.95, scale = False, is_training = True)
    C3 = tf.nn.relu(C3)
    C3 = conv_1s(C3, WC_reduction_3['_33'])
    C3 = batch_normalization(C3, 0.95, scale = False, is_training = True)
    C3 = C3 + biases_3['b_2']
    C3 = tf.nn.relu(C3)

    C5 = conv_1s(input_layer, WC_reduction_5['red'])
    C5 = C5 + biases_5['b_1']
    C5 = batch_normalization(C5, 0.95, scale = False, is_training = True)
    C5 = tf.nn.relu(C5)
    C5 = conv_1s(C5, WC_reduction_5['_33'])
    C5 = C5 + biases_5['b_2']
    C5 = batch_normalization(C5, 0.95, scale = False, is_training = True)
    C5 = tf.nn.relu(C5)

    C1 = conv_1s(input_layer, WC_1['_11'])
    C1 = C1 + biases_1['b_1']
    C1 = batch_normalization(C1, 0.95, scale = False, is_training = True)
    C1 = tf.nn.relu(C1)

    C_pool = conv_1s(input_layer, WC_pool['_11'])
    C_pool = C_pool + biases_pool['b_1']
    C_pool = batch_normalization(C_pool, 0.95, scale = False, is_training = True)
    C_pool = tf.nn.relu(C_pool)

    C = tf.concat([C1, C3, C5, C_pool], 3)

    return C

def Fully_connect_conv(input_layer, units, activation = tf.nn.relu, weights_name = None, biases_name = None):
    initializer = tf.contrib.layers.xavier_initializer()
    flattened = tf.contrib.layers.flatten(input_layer)
    print('fshape'+str(flattened.shape))
    shapes = flattened.shape[1]
    weights = tf.Variable(initializer([int(shapes), units]), name = weights_name)
    biases = tf.Variable(tf.zeros([units]), name = biases_name)
    FC = tf.matmul(flattened, weights)+biases
    FC = activation(FC)
    return FC

def Fully_connect_dense(input_layer, units, activation = tf.nn.relu, weights_name = None, biases_name = None):
    initializer = tf.contrib.layers.xavier_initializer()
    shapes = input_layer.shape[1]
    weights = tf.Variable(initializer([int(shapes), units]), name = weights_name)
    biases = tf.Variable(tf.zeros([units]), name = biases_name)
    if activation != None:
        FC = tf.matmul(input_layer, weights) + biases
        FC = activation(FC)
    else: FC = tf.matmul(input_layer, weights) + biases
    return FC

def Average_pool(x, pool_size = 5, stride = 3):
    return tf.nn.avg_pool(x, ksize = [1, pool_size, pool_size, 1], strides = [1, stride, stride, 1], padding = 'SAME')



def batch_normalization(x, decay, scale = True, is_training = True):
    normed =  tf.contrib.layers.batch_norm(x,
                                           decay=decay,
                                           scale=scale,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS,
                                           is_training=True)    
    return normed







































