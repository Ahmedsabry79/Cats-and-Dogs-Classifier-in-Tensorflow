import cv2
import os
import random
import numpy, time, pickle
from scipy import ndimage
from scipy.misc import imresize
from numpy import flip

path = r'E:\Information Technology\Machine Learning & Deep Learning\Udemy\Deep Learning Course\10 Building a CNN\Data\Convolutional_Neural_Networks\dataset'
img_size= 224
def Dataset_generator(path):
    training_set = []
    test_set = []
    for file in os.listdir(path):
        if file == 'training_set':
            print('started the training_set')
            np = os.path.join(path, file)
            for nfile in os.listdir(np):
                if nfile == 'cats':
                    print('Started in cats file')
                    class_ = [1.]
                    nnp = os.path.join(np, nfile)
                    i = 0
                    for img in os.listdir(nnp):
                        if not '.DS_Store' in img:
                            img_path = os.path.join(nnp, img)
                            new_img = ndimage.imread(img_path, mode = 'RGB')
                            new_img = imresize((new_img), (img_size, img_size))
                            new_img = (new_img-255/2)/255
                            flipped = flip(new_img, 1)
                            set_ = [new_img, class_]
                            fset_ = [flipped, class_]
                            training_set.append(set_)
                            training_set.append(fset_)
                            i+= 1
                            if i in range(100, 100000, 100):
                                print(str(i), '  images are done out of ', str(len(os.listdir(nnp))-1))
                if nfile == 'dogs':
                    print('Started in dogs file')
                    class_ = [0.]
                    nnp = os.path.join(np, nfile)
                    i = 0
                    for img in os.listdir(nnp):
                        if not '.DS_Store' in img:
                            img_path = os.path.join(nnp, img)
                            new_img = ndimage.imread(img_path, mode = 'RGB')
                            new_img = imresize(new_img, (img_size, img_size))
                            new_img = (new_img-255/2)/255
                            flipped = flip(new_img, 1)
                            set_ = [new_img, class_]
                            fset_ = [flipped, class_]
                            training_set.append(set_)
                            training_set.append(fset_)
                            i+= 1
                            if i in range(100, 100000, 100):
                                print(str(i), '  images are done out of ', str(len(os.listdir(nnp))-1))
        if file == 'test_set':
            print('started the test_set')
            np = os.path.join(path, file)
            for nfile in os.listdir(np):
                if nfile == 'cats':
                    print('Started in cats file')
                    class_ = [1.]
                    nnp = os.path.join(np, nfile)
                    i = 0
                    for img in os.listdir(nnp):
                        if not '.DS_Store' in img:
                            img_path = os.path.join(nnp, img)
                            new_img = ndimage.imread(img_path, mode = 'RGB')
                            new_img = imresize(new_img, (img_size, img_size))
                            new_img = (new_img-255/2)/255
                            set_ = [new_img, class_]
                            test_set.append(set_)
                            i+= 1
                            if i in range(100, 10000, 100):
                                print(str(i), '  images are done out of ', str(len(os.listdir(nnp))-1))
                if nfile == 'dogs':
                    print('Started in dogs file')
                    class_ = [0.]
                    nnp = os.path.join(np, nfile)
                    i = 0
                    for img in os.listdir(nnp):
                        if not '.DS_Store' in img:
                            img_path = os.path.join(nnp, img)
                            new_img = ndimage.imread(img_path, mode = 'RGB')
                            new_img = imresize(new_img, (img_size, img_size))
                            new_img = (new_img-255/2)/255
                            set_ = [new_img, class_]
                            test_set.append(set_)
                            i+= 1
                            if i in range(100, 100000, 100):
                                print(str(i), '  images are done out of ', str(len(os.listdir(nnp))-1))
    random.shuffle(training_set)
    random.shuffle(test_set)
    random.shuffle(training_set)
    random.shuffle(test_set)
    training_set = numpy.array(training_set)
    test_set = numpy.array(test_set)

    x_train = [training_set[i][0] for i in range(len(training_set))]
    x_train = numpy.array([i for i in x_train], numpy.float32)

    y_train = [training_set[i][1] for i in range(len(training_set))]
    y_train = numpy.array([i for i in y_train], numpy.float32)

    x_test = [test_set[i][0] for i in range(len(test_set))]
    x_test = numpy.array([i for i in x_test], numpy.float32)

    y_test = [test_set[i][1] for i in range(len(test_set))]
    y_test = numpy.array([i for i in y_test], numpy.float32)

##    with open('data_xtrain1.pickle', 'wb') as data:
##        Data = {'x_train': x_train[:2000]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_xtrain2.pickle', 'wb') as data:
##        Data = {'x_train': x_train[2000:4000]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_xtrain3.pickle', 'wb') as data:
##        Data = {'x_train': x_train[4000:6000]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_xtrain4.pickle', 'wb') as data:
##        Data = {'x_train': x_train[6000:8000]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_xtrain5.pickle', 'wb') as data:
##        Data = {'x_train': x_train[8000:10000]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_xtrain6.pickle', 'wb') as data:
##        Data = {'x_train': x_train[10000:12000]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_xtrain7.pickle', 'wb') as data:
##        Data = {'x_train': x_train[12000:14000]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_xtrain8.pickle', 'wb') as data:
##        Data = {'x_train': x_train[14000:]}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##        
##        
##    with open('data_xtest.pickle', 'wb') as data:
##        Data = {'x_test': x_test}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##        
##    with open('data_xtest.pickle', 'wb') as data:
##        Data = {'x_test': x_test}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytest.pickle', 'wb') as data:
##        Data = {'y_test': y_test}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytest.pickle', 'wb') as data:
##        Data = {'y_test': y_test}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##    
##    with open('data_ytrain1.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytrain2.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytrain3.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytrain4.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytrain5.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytrain6.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytrain7.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')
##
##    with open('data_ytrain8.pickle', 'wb') as data:
##        Data = {'y_train': y_train}
##        pickle.dump(Data, data, 0)
##    print('file is made')

    
    
    return x_train, y_train, x_test, y_test

#Dataset_generator(path)




