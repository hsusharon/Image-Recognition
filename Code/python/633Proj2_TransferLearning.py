#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import torch

from resnet_pytorch import ResNet
from skimage.transform import resize
from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers


# In[5]:


#load data with function(define function)
train_dir = "C:/Users/USER/Desktop/633 Proj2/data/Monkey Database/training/training/"
test_dir =  "C:/Users/USER/Desktop/633 Proj2/data/Monkey Database/validation/validation/"
from tqdm import tqdm

def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    
    print("Fetching data from:", folder);
    for folderName in os.listdir(folder):
        
        if not folderName.startswith('.'):
            if folderName in ['n0']:
                label = 0
            elif folderName in ['n1']:
                label = 1
            elif folderName in ['n2']:
                label = 2
            elif folderName in ['n3']:
                label = 3
            elif folderName in ['n4']:
                label = 4
            elif folderName in ['n5']:
                label = 5
            elif folderName in ['n6']:
                label = 6
            elif folderName in ['n7']:
                label = 7
            elif folderName in ['n8']:
                label = 8
            elif folderName in ['n9']:
                label = 9
            else:
                label = 10
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (150, 150, 3))
                    #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y


# In[ ]:


## load data
X_train, y_train = get_data(train_dir)
X_test, y_test= get_data(test_dir)
print("Data Fetch complete");


# In[4]:


print("Shape of X_train:", np.shape(X_train));
print("Shape of y_train:", np.shape(y_train));
print("Shape of X_test:", np.shape(X_test));
print("Shape of y_test:", np.shape(y_test));


# In[9]:


get_ipython().run_cell_magic('time', '', '##reshape data into one color\ncolor = 3;\ntraining_data = 1098*[150*[150*[0]]];\nfor i in range(1098):\n    for j in range(150):\n        for k in range(150):\n            total = 0;\n            for l in range(color):\n                total = total + X_train[i][j][k][l];\n            training_data[i][j][k] = total/color;\n\ntesting_data = 272*[150*[150*[0]]];\nfor i in range(272):\n    for j in range(150):\n        for k in range(150):\n            total = 0;\n            for l in range(color):\n                total = total + X_test[i][j][k][l];\n            testing_data[i][j][k] = total/color;\n\nprint("Shape of X_train:", np.shape(training_data));\nprint("Shape of X_test:", np.shape(testing_data));\n    ')


# In[10]:


training_data = np.array(training_data);
testing_data = np.array(testing_data);
print(training_data[0]);


# In[106]:


##normalize the input data
#training_data = tf.keras.utils.normalize(training_data, axis=1);
#testing_data = tf.keras.utils.normalize(testing_data, axis=1);

##construct a 5 layer CNN model
model_cnn = tf.keras.models.Sequential()
model_cnn.add(tf.keras.layers.Flatten()) ##input layer
##128 nurons per layer, activation function
model_cnn.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))  
model_cnn.add(tf.keras.layers.Dense(512, activation=tf.nn.relu)) 
model_cnn.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model_cnn.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) 

model_cnn.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model_cnn.fit(training_data, y_train, epochs=10);


# In[63]:


val_loss, val_acc = model.evaluate(testing_data,y_test)
print(val_loss, val_acc)


# In[64]:


##pop the last layer 
model_cnn.summary();
model_cnn.pop();  ##discard the last layer
model_cnn.summary();  

for layer in model_cnn.layers:  ##we freeze the previously trained CNN model
    layer.trainable = False;
model_cnn.summary();


# In[ ]:


##truncate to the pretrained model provided by python

##load pretrained model VGG
ResNet = tf.keras.applications.resnet50.ResNet50();
#ResNet.summary();

last_layer = ResNet.get_layer('avg_pool') #the name of the last layer you want from the model
last_output = last_layer.output
input_l = ResNet.input
model_resnet = tf.keras.Model(input_l, last_output)

fc = Dense(10,activation='softmax')(model_resnet)
model_resnet = Model(inputs=model_resnet.input, outputs=fc2)

model_resnet.summary();  ##delete the last layer 
#model_resnet.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
#model_resnet.summary();

#model_resnet.summary();
# model_inception = tf.keras.applications.inception_v3.InceptionV3(
#     weights=None,
#     input_shape=(128), classes=10,
#     classifier_activation='softmax');
# model_inception.summary();
#print(model_resnet.eval());
#model_resnet.summary();
#merge_layer = tf.keras.layers.Concatenate(model_cnn.layers[0], model_resnet.layers[0]);


# In[ ]:




