#!/usr/bin/env python
# coding: utf-8

# In[37]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[38]:


##pass int the dataset from mnist
mnist = tf.keras.datasets.mnist  # 0-9 hand-written digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

##we normalize the data so that the neural networks are trained better
#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[39]:


##visialize the dataset we are dealing with
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print('Label of the training data[0]:', y_train[0])


# In[40]:


##build up the CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) ##input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
##128 nurons per layer, activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) 

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)


# In[41]:


##check if the model overfits
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)


# In[42]:


##we save the model
model.save('model_saved')
new_model = tf.keras.models.load_model('model_saved')


# In[43]:


##predict the data with test dataset
predictions = new_model.predict([x_test])
print('Label of the prediction:', np.argmax(predictions[1])) ##the prediction of the 1 value of the testing dataset
plt.imshow(x_test[1], cmap = plt.cm.binary) ##to plot the testing data to verify
plt.show()


# In[ ]:




