
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import argparse
# sess = tf.InteractiveSession()
from sklearn.metrics import roc_auc_score
from keras.metrics import categorical_accuracy as accuracy


# In[2]:


def parser_tfrecord(record):
    features = tf.parse_single_example(record,
                                       features={
                                           "dvc": tf.FixedLenFeature([], tf.string),
                                           "height": tf.FixedLenFeature([], tf.int64),
                                           "width": tf.FixedLenFeature([], tf.int64),
                                           "label": tf.FixedLenFeature([], tf.string),
                                           "app_use": tf.FixedLenFeature([], tf.string),
                                       })
    
    app_use = tf.decode_raw(features['app_use'], tf.float32)
    app_use = tf.reshape(app_use, shape=[31, 7120, 1])

    label = tf.decode_raw(features['label'], tf.int64)
    label = tf.reshape(label, shape=[2])

    return label, app_use


def read_shuffle_feature(batch_size, shuffle_buffer, epochs, filename):
#     filename = os.path.join(FLAGS.directory, name + '.tfrecords')
#     filename = 'hdfs://ns-hf/user/mzzhao/lsi.tfrecords'
    filename = 
    dataset = tf.data.TFRecordDataset(filename).map(parser_tfrecord).shuffle(shuffle_buffer).batch(batch_size).repeat(epochs)

    iterator = dataset.make_initializable_iterator()
    #label, app_use = iterator.get_next()
          
    return iterator

batch_size = 64
shuffle_buffer = 1000
num_epochs = 1
iterator = read_shuffle_feature(batch_size, shuffle_buffer, num_epochs, './train_month_nL.tfrecords')

test_iterator = read_shuffle_feature(batch_size, shuffle_buffer, num_epochs, './newTest_sample_month.tfrecords')


# In[4]:


input_layer = tf.keras.layers.Input(shape = (31, 7120, 1))  # 实例化一个Keras张量


# In[5]:


# # 卷积层数设置为2
# conv2d_layer_2 = tf.keras.layers.Conv2D(filters = 64,
#                                        kernel_size = (2, 1780),
#                                        strides = (1,1780),
#                                        padding = 'same',
#                                        activation = 'relu')(input_layer)
# pool_layer_2 = tf.keras.layers.MaxPooling2D(pool_size = (3,1))(conv2d_layer_2)
# drop_layer_2 = tf.keras.layers.Dropout(0.5)(pool_layer_2)
# flat_layer_2 = tf.keras.layers.Flatten()(drop_layer_2)


# In[6]:


# #  卷积层数设置为3
# conv2d_layer_3 = tf.keras.layers.Conv2D(filters = 64,
#                                        kernel_size = (3, 1780),
#                                        strides = (1,1780),
#                                        padding = 'same',
#                                        activation = 'relu')(input_layer)
# pool_layer_3 = tf.keras.layers.MaxPooling2D(pool_size = (3,1))(conv2d_layer_3)
# drop_layer_3 = tf.keras.layers.Dropout(0.5)(pool_layer_3)   #防止过拟合
# flat_layer_3 = tf.keras.layers.Flatten()(drop_layer_3)   #压平输入


# In[7]:


#  卷积层数设置为4
conv2d_layer_4 = tf.keras.layers.Conv2D(filters = 64,
                                       kernel_size = (4, 7120),
                                       strides = (1,7120),
                                       padding = 'same',
                                       activation = 'relu')(input_layer)
pool_layer_4 = tf.keras.layers.MaxPooling2D(pool_size = (3,1))(conv2d_layer_4)
drop_layer_4 = tf.keras.layers.Dropout(0.5)(pool_layer_4)
flat_layer_4 = tf.keras.layers.Flatten()(drop_layer_4)


# In[8]:


#  卷积层数设置为5
conv2d_layer_5 = tf.keras.layers.Conv2D(filters = 64,
                                       kernel_size = (5, 7120),
                                       strides = (1,7120),
                                       padding = 'same',
                                       activation = 'relu')(input_layer)
pool_layer_5 = tf.keras.layers.MaxPooling2D(pool_size = (3,1))(conv2d_layer_5)
drop_layer_5 = tf.keras.layers.Dropout(0.5)(pool_layer_5)
flat_layer_5 = tf.keras.layers.Flatten()(drop_layer_5)


# In[9]:


#  卷积层数设置为6
conv2d_layer_6 = tf.keras.layers.Conv2D(filters = 64,
                                       kernel_size = (6, 7120),
                                       strides = (1,7120),
                                       padding = 'same',
                                       activation = 'relu')(input_layer)
pool_layer_6 = tf.keras.layers.MaxPooling2D(pool_size = (3,1))(conv2d_layer_6)
drop_layer_6 = tf.keras.layers.Dropout(0.5)(pool_layer_6)
flat_layer_6 = tf.keras.layers.Flatten()(drop_layer_6)


# In[10]:


#  卷积层数设置为7
conv2d_layer_7 = tf.keras.layers.Conv2D(filters = 64,
                                       kernel_size = (7, 7120),
                                       strides = (1,7120),
                                       padding = 'same',
                                       activation = 'relu')(input_layer)
pool_layer_7 = tf.keras.layers.MaxPooling2D(pool_size = (3,1))(conv2d_layer_7)
drop_layer_7 = tf.keras.layers.Dropout(0.5)(pool_layer_7)
flat_layer_7 = tf.keras.layers.Flatten()(drop_layer_7)


# In[11]:


flat_layer = tf.keras.layers.Concatenate()([flat_layer_4, flat_layer_5, flat_layer_6, flat_layer_7])  # 将4层网络输出拉平，每层960，共3840


# In[12]:


output_layer = tf.keras.layers.Dense(2, activation = 'softmax')(flat_layer)  #  output arrays of shape (*, 1)
model = tf.keras.models.Model(inputs = input_layer, outputs = output_layer)
model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy']) 
model.compile


# In[13]:


print(model.summary())


# In[17]:


X_train = []
Y_train = []
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
label_batch, app_batch = iterator.get_next()

x_test = []
y_test = []
test_label_batch, test_app_batch = test_iterator.get_next()

with tf.Session(config=sess_config) as mon_sess:
    print('Session stared.')
    mon_sess.run(tf.global_variables_initializer())
    mon_sess.run(iterator.initializer)

    while True:
        try:
            x_batch, y_batch = mon_sess.run([app_batch, label_batch])

            x_batch = x_batch.reshape(-1, 31, 7120, 1)
            y_batch = y_batch.reshape(-1, 2)
            X_train.extend(x_batch)
            Y_train.extend(y_batch)

#                 _, loss_value, global_step_value, y_pd = mon_sess.run(
#                     [train_op, loss, global_step, y_pred], feed_dict={x: app_batch, y_: y_batch})
#                 print('%')
#                 print(y_pd)
#                 print(loss_value)
        except tf.errors.OutOfRangeError:
            print("Out of Range!")
            break
            
    mon_sess.run(test_iterator.initializer)
    while True:
        try:
            x_batch, y_batch = mon_sess.run([test_app_batch, test_label_batch])

            x_batch = x_batch.reshape(-1, 31, 7120, 1)
            y_batch = y_batch.reshape(-1, 2)
            x_test.extend(x_batch)
            y_test.extend(y_batch)

#                 _, loss_value, global_step_value, y_pd = mon_sess.run(
#                     [train_op, loss, global_step, y_pred], feed_dict={x: app_batch, y_: y_batch})
#                 print('%')
#                 print(y_pd)
#                 print(loss_value)
        except tf.errors.OutOfRangeError:
            print("Out of Range!")
            break


# In[ ]:


model.fit(np.array(X_train), 
          np.array(Y_train),
          epochs = 2, 
          batch_size = 128)

y_pre = model.predict(np.array(x_test))
print('test_auc',roc_auc_score(np.array(y_test),y_pre))


# In[ ]:


test_acc_value = accuracy(np.array(y_test), y_pre)
with tf.Session() as sess:
    print(np.mean(sess.run(test_acc_value)))

