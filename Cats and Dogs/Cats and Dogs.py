import GoogleNet, Image_Generator
import tensorflow as tf
import numpy as np

path = r'D:\Information Technology\Deep Learning\Data\Convolutional_Neural_Networks\dataset'

x_train, y_train, x_test, y_test = Image_Generator.Dataset_generator(path)

tf.reset_default_graph()
x = tf.placeholder(tf.float32,[None, 224, 224, 3], name =  'x')
y = tf.placeholder(tf.float32, [None, 1], name = 'y')
is_training = tf.placeholder(tf.bool, name = 'is_training')

epochs = 50
batch_size = 32
lr = 0.0001

n_per_epoch = int(len(x_train)//batch_size)
test_accuracies = []
train_accuracies = []

def construct(x):
    Layer_1 = GoogleNet.Inception_Layer_1(x, is_training, filter_size = 7, channels = 3, n_filters = 64,
                                conv_stride = 2, pool_stride = 2, pool_ksize = 3, weights_name = 'L_1', biases_name = 'b_1')
    
    Layer_2 = GoogleNet.Inception_Layer_2(Layer_1, is_training, 64, filter_size = 3,
                                reduce_to_filters = 16, output_filters = 64,
                                conv_stride = 1, pool_stride = 2, pool_ksize = 3, weights_name = 'L_2', biases_name = 'b_2')

    Layer_3_a = GoogleNet.Inception_Layer(Layer_2, is_training, 64,
                                reduce_to_filters_3 = 16, output_filters_3 = 32, 
                                reduce_to_filters_5 = 8, output_filters_5 = 16, output_filters_1 = 32,
                                output_filters_pool = 16, conv_stride = 1, pool_stride = 1, pool_ksize = 3, weights_name = 'L_3a', biases_name = 'b_3a')

    Layer_3_b = GoogleNet.Inception_Layer(Layer_3_a, is_training, 96,
                                reduce_to_filters_3 = 32, output_filters_3 = 64, 
                                reduce_to_filters_5 = 16, output_filters_5 = 32, output_filters_1 = 32,
                                output_filters_pool = 16, conv_stride = 1, pool_stride = 1, pool_ksize = 3, weights_name = 'L_3c', biases_name = 'b_3c')

    Pooled_layer = GoogleNet.Maxpool_2s_33(Layer_3_b)
    
    Layer_4 = GoogleNet.Inception_Layer(Pooled_layer, is_training, 144,
                              reduce_to_filters_3 = 32, output_filters_3 = 96, 
                              reduce_to_filters_5 = 16, output_filters_5 = 32, output_filters_1 = 64,
                              output_filters_pool = 32, conv_stride = 1, pool_stride = 1, pool_ksize = 3, weights_name = 'L_4', biases_name = 'b_4')
        
    Avg_pool_layer = GoogleNet.Average_pool(Layer_4, 3, 1)

    
    FC1 = GoogleNet.Fully_connect_conv(Avg_pool_layer, 250, activation = tf.nn.relu, weights_name = 'FC1_W', biases_name = 'FC1_b')
    
    DFC1 = tf.nn.dropout(FC1, 0.4)
    
    FC2 = GoogleNet.Fully_connect_dense(DFC1, 250, activation = tf.nn.relu, weights_name = 'FC2_W', biases_name = 'FC2_b')
    
    DFC2 = tf.nn.dropout(FC2, 0.4)
    
    FC3 = GoogleNet.Fully_connect_dense(DFC2, 50, activation = tf.nn.relu, weights_name = 'FC3_W', biases_name = 'FC3_b')
    
    FCE = GoogleNet.Fully_connect_dense(FC3, 1, activation = None, weights_name = 'FCE_W', biases_name = 'FCE_b')
    
    return FCE



def evaluate(x, lr):
    predictions = construct(x)
    sigmoided = tf.nn.sigmoid(predictions)
    corr = tf.equal(tf.round(sigmoided), y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = predictions, labels = y))

    op = tf.train.AdamOptimizer(lr).minimize(cost)
    opts = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.group([op, opts])
    
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.333
    config.gpu_options.allow_growth = True   

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            total_correct_train = []
            epoch_loss = 0
            start = 0
            end = batch_size
            for it in range(n_per_epoch):
                ex = x_train[start:end]
                ey = y_train[start:end]
                start += batch_size
                end += batch_size
                print('Batch number '+ str(it+1), ' ended out of ', str(n_per_epoch), ' Batches')
                _, c = sess.run([optimizer, cost], feed_dict = {'x:0': ex, 'y:0': ey, is_training: True})
                correct = corr.eval(feed_dict = {'x:0': ex, 'y:0': ey, is_training: False})
                total_correct_train.append(correct)
                epoch_loss += c


            total_correct_test = []

            n_iters = len(x_test)//20
            start = 0
            end = 20
            for i in range(n_iters):
                test_batch_x = x_test[start: end]
                test_batch_y = y_test[start: end]
                start += 20
                end += 20
                acc = corr.eval(feed_dict = {'x:0': test_batch_x, 'y:0': test_batch_y, is_training: False})
                total_correct_test.append(acc)


            accuracy_train = np.mean(np.array(total_correct_train).ravel())
            accuracy_test = np.mean(np.array(total_correct_test).ravel())
            print('epoch', epoch+1, 'is done out of '+str(epochs)+' epochs and the train accuracy is: ', accuracy_train, ' test accuracy is: ', accuracy_test, 'Loss is: ', epoch_loss)
            
evaluate(x, lr)










