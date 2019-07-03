import numpy as np
import os
import cv2
import random
from scipy import ndimage


## Importing and loading the data:

mask_path = r'E:\Information Technology\Machine Learning & Deep Learning\Projects\TGS Salt Identification Challenge\Data\Train\masks'
train_path = r'E:\Information Technology\Machine Learning & Deep Learning\Projects\TGS Salt Identification Challenge\Data\Train\images'

def training_data_generator(mask_path, train_path, img_size):
    Dataset = []
    Test_Dataset = []
    
    N = 4000
    n = 100
    Images = os.listdir(mask_path)
    random.shuffle(Images)
    test_images = Images[:1000]
    train_images = Images[1000:]
    for img in train_images:
        cycle = []
        if train_images.index(img) in range(0, 4000, 100):
            print(n, ' Images are preprocessed out of 4000 Images')
            n = n + 100
            
        ## Reading the Image and its Corresponding masks:
        original_img = (ndimage.imread(os.path.join(train_path, img))-255.0/2)/255
        mask_img = (ndimage.imread(os.path.join(mask_path, img))-255.0/2)/255
        

        ## Resizing the original Image to the needed size:
        original_img_256 = cv2.resize(original_img, (img_size, img_size))
        cycle.append([original_img_256, mask_img])
        
        ## Flipping 1:
        flipped_original_256 = cv2.flip(original_img_256, 1)
        flipped_mask = cv2.flip(mask_img, 1)
        cycle.append([flipped_original_256, flipped_mask])

        ## Flipping 2:
        flipped_original_256 = cv2.flip(original_img_256, 0)
        flipped_mask = cv2.flip(mask_img, 0)
        cycle.append([flipped_original_256, flipped_mask])

        ## Flipping 3:
        flipped_original_256 = cv2.flip(original_img_256, -1)
        flipped_mask = cv2.flip(mask_img, -1)
        cycle.append([flipped_original_256, flipped_mask])

        ## 90 degrees rotation:
        rotated_original_256 = cv2.rotate(original_img_256, cv2.ROTATE_90_CLOCKWISE)
        rotated_mask = cv2.rotate(mask_img, cv2.ROTATE_90_CLOCKWISE)
        cycle.append([rotated_original_256, rotated_mask])

        ## Random Crops 1:
        Rx = random.randint(0, 64)
        Ry = random.randint(0, 64)

        original_img_256 = cv2.resize(original_img, (256, 256))
        mask_img_256 = cv2.resize(mask_img, (256, 256))
        
        C_original_img = original_img_256[Ry:Ry+192, Rx: Rx+192]
        C_mask_img = mask_img_256[Ry:Ry+192, Rx: Rx+192]

        C_original_img_256 =cv2.resize(C_original_img, (img_size, img_size))
        C_mask_img_101 = cv2.resize(C_mask_img, (101, 101))

        cycle.append([C_original_img_256, C_mask_img_101])

        ## Random Crops 2:
        Rx = random.randint(0, 128)
        Ry = random.randint(0, 128)

        original_img_256 = cv2.resize(original_img, (256, 256))
        mask_img_256 = cv2.resize(mask_img, (256, 256))
        
        C_original_img = original_img_256[Ry:Ry+128, Rx: Rx+128]
        C_mask_img = mask_img_256[Ry:Ry+128, Rx: Rx+128]

        C_original_img_256 =cv2.resize(C_original_img, (img_size, img_size))
        C_mask_img_101 = cv2.resize(C_mask_img, (101, 101))

        cycle.append([C_original_img_256, C_mask_img_101])

        for r in cycle:
            ## Obtaining the labels of the image from the mask
            img_list = [r[1][:, i] for i in range(r[1].shape[1])]
            img_list = [j for i in img_list for j in i]
            labels = np.array(img_list)
            labels[labels > 0] = 1
            labels = labels.tolist()
            Dataset.append([np.array(r[0], dtype =np.float16), np.array(labels, dtype = np.float16)])
        
    random.shuffle(Dataset)
    random.shuffle(Dataset)
    random.shuffle(Dataset)
    
    X_train = [i[0] for i in Dataset]
    y_train = [i[1] for i in Dataset]
    
    
    ## Test Data Preparation:
    for img in test_images:
        cycle = []
        if test_images.index(img) in range(0, 4000, 100):
            print(n, ' Images are preprocessed out of 4000 Images')
            n = n + 100
            
        ## Reading the Image and its Corresponding masks:
        original_img = (ndimage.imread(os.path.join(train_path, img))-255.0/2)/255
        mask_img = (ndimage.imread(os.path.join(mask_path, img), 0)-255.0/2)/255
        
        ## Resizing the original Image to the needed size:
        original_img_256 = cv2.resize(original_img, (img_size, img_size))
        cycle.append([original_img_256, mask_img])
        
        for r in cycle:
            ## Obtaining the labels of the image from the mask
            img_list = [r[1][:, i] for i in range(r[1].shape[1])]
            img_list = [j for i in img_list for j in i]
            labels = np.array(img_list)
            labels[labels > 0] = 1
            labels = labels.tolist()
            Test_Dataset.append([np.array(r[0], dtype =np.float16), np.array(labels, dtype = np.float16)])
    
    X_test = [i[0] for i in Test_Dataset]
    y_test = [i[1] for i in Test_Dataset]

    return np.array(X_train, dtype = np.float16), np.array(y_train, dtype = np.float16), np.array(X_test, dtype = np.float16), np.array(y_test, dtype = np.float16)


x_train, y_train, x_test, y_test = training_data_generator(mask_path, train_path, 224)


def convert_to_image(label):
    label_img = []
    for i in range(0, len(label), 101):
        start = 0
        end = 101
        row = [label[start: end]]
        label_img.append(row)
        start += 101
        end += 101
    return label_img





        



























