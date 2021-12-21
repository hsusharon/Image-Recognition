#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics


# In[2]:


##load the training and testing data
mnist = tf.keras.datasets.mnist  # 0-9 hand-written digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

##SVM are only able to distinguish two different classes
##we distinguish digits that are smaller or greater than 5
label_train = [0 for i in range(len(y_train))];
for index in range(len(y_train)):
    if y_train[index] < 5:
        label_train[index] = 0;
    else:
        label_train[index] = 1;
    #print(label_train[index])

label_test = [0 for i in range(len(y_test))];
for index in range(len(y_test)):
    if y_test[index] < 5:
        label_test[index] = 0;
    else:
        label_test[index] = 1;

print(len(x_train),len(y_train));
print(len(x_train[0]))
print(len(x_test),len(y_test));


# In[3]:


##reshape data

data_reshape_train = 60000*[784*[0]]; ##28*28=784
data_reshape_test = 10000*[784*[0]];
for index in range(len(x_train)):
    data_reshape_train[index] = np.reshape(x_train[index],(784));

for index in range(len(x_test)):
    data_reshape_test[index] = np.reshape(x_test[index], (784));

print(np.shape(data_reshape_train))


# In[6]:


##apply PCA
pca = PCA(n_components=1); ##svd_solver='arpack'
data_training = pca.fit_transform(data_reshape_train);
data_testing = pca.fit_transform(data_reshape_test);

print(np.shape(data_training));
print(np.shape(data_testing));

colormap = np.array(['red', 'blue'])

y = 60000*[0];
plt.scatter(data_training, y, c=colormap[label_train])
plt.show()
#print(len(data_training));


# In[ ]:


##train svm (linear)
cls = svm.SVC(kernel="linear");
cls.fit(data_training, label_train);
pred = cls.predict(data_testing);
print(pred[0]);

print("Linear accuracy:", metrics.accuracy_score(label_test, y_pred = pred))


# In[ ]:


##train svm (poly)
cls = svm.SVC(kernel="poly");
cls.fit(data_training, label_train);
pred = cls.predict(data_testing);
print(pred[0]);

print("Poly accuracy:", metrics.accuracy_score(y_test, y_pred = pred))


# In[ ]:


##train svm (rbf)
cls = svm.SVC(kernel="rbf");
cls.fit(data_training, y_train);
pred = cls.predict(data_testing);
print(pred[0]);

print("RBF accuracy:", metrics.accuracy_score(y_test, y_pred = pred))
