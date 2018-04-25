
# coding: utf-8

# In[1]:


'''
将用户的app安装列表转化为最后一天（1*7000）和一个月（31*7000）的数据格式，然后转化为tfrecord
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import pickle as pk
import h5py
import argparse


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_app_list():
    #读app列表，可选数量为11万和7000
    app_list=[]
    f1 = open('app_list.txt',encoding='utf-8')
    for i in f1.readlines():
        line =i.strip()
        app_list.append(line)
    f1.close()
    print(app_list[:3])
    return app_list


def gen_data(periods, n_feature, app_info,time, app_list, is_last_row):
    #json_info = str(app_info)
    last_time = time
    #print('time',last_time)
    #print('info',app_info)
    #print(json_info[:2])
    
    df = pd.DataFrame(np.zeros(shape=(periods, n_feature)), columns=app_list)
    for rec in eval(app_info):
        #print(type(rec))
        date_list = pd.Series(np.zeros(periods), index=pd.date_range(end=last_time, periods=periods))
        for date in rec['load_info']:
            date_list[date] = 1

        df[rec['app_name'].strip()] = date_list.values
    '''
    is_last_row=True，表示只取最后一天的安装列表
    is_last_row=False，表示取一个月的安装列表，且需要注释掉下面一行代码！！！！！！！！
    
    '''
    df = df.loc[~(df == 0).all(axis=1)]
    
    if is_last_row:
        #newTest中有空值，所以下面代码会报错，加try后依然有错，data未声明，需要重新赋值
        try:
            data = df.iloc[-1]
        except IndexError:
            print(df)
    else:
        data = df

    #print('gen_data is over!')
    return data


def convert_to(data_set, name):
    #file = h5py.File(data_set, 'r')
    f = open(data_set)
    dvc_list = [];time = [];app_info = [];label_list = []
    for i in f.readlines():
        line = i.strip().split('\t')
        #line[3] = eval(line[3])
        dvc_list.append(line[1])
        time.append(line[2])
        app_info.append(line[3])
        label_list.append(line[4])
    
    x_train = {}
    # read label list
    #label_dict, dvc_list = get_label_list()
    app_list = get_app_list()

    i = 0
    for index,dvc in enumerate(dvc_list):
        app = np.array(gen_data(periods=31, n_feature=len(app_list),app_info=app_info[index],time=time[index], app_list=app_list, is_last_row=True))
        i += 1
        if i == 1:
            print(app)
            print(len(app))
            print(type(app))
            #pd.DataFrame(app).to_csv('forTest.csv',index=0)
        #if i >= 100:
        #   break

        x_train[dvc] = app

    f.close()
    #     x_train = np.array(x_train)
    num_examples = len(x_train)
    print('测试集长度',num_examples)

    app_info = np.array(x_train[list(x_train.keys())[0]])
    print(app_info)
    rows = 1
    print(rows)

    cols = app_info.shape[0]
    print(cols)

    filename = name + '.tfrecords'
    print('Writing', filename)

    writer = tf.python_io.TFRecordWriter(filename)
    index = 0
    for key, value in x_train.items():
        #         app_use = np.random.randn(5, 500)
        #         print(len(app_use))
        app_use = np.array(value)
        #label = float(label_list[dvc_list.index(key)])会报以下错误
        #ValueError: invalid literal for int() with base 10: '0.0'  后将int()改为int(float())
        label = int(float(label_list[dvc_list.index(key)]))
        label = np.concatenate(
            [(1 - np.array(label)).reshape(-1, 1).tolist(), np.array(label).reshape(-1, 1).tolist()], axis=1)
        label = np.array(label.tolist()).astype(np.int64)
        #         print(app_use.dtype)
        app_use = app_use.astype(np.float32)
        dvc = bytes(key, encoding="utf8")
        
        print('dvc:',dvc)
        print('label:',label)
        print('app_use',app_use)
 
        example = tf.train.Example(features=tf.train.Features(feature={
            'dvc': _bytes_feature(dvc),
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'label': _bytes_feature(label.tostring()),
            'app_use': _bytes_feature(app_use.tostring())
        }))
        writer.write(example.SerializeToString())
        index += 1
        if index == 2:
            print(label)
            print(dvc)
            print(app_use)

    writer.close()
    f.close()
    print('Done')
'''
def read_and_decode(name):
    #     for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    #         example = tf.train.Example()
    #         example.ParseFromString(serialized_example)

    #         image = example.features.feature['app_use'].float_list.value

    #         # 可以做一些预处理之类的
    #         print(image)

    reader = tf.TFRecordReader()
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')

    # Even when reading in multiple threads, share the filename
    filename_queue = tf.train.string_input_producer([filename])

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "dvc": tf.FixedLenFeature([], tf.string),
                                           "height": tf.FixedLenFeature([], tf.int64),
                                           "width": tf.FixedLenFeature([], tf.int64),
                                           "label": tf.FixedLenFeature([], tf.int64),
                                           "app_use": tf.FixedLenFeature([], tf.string),
                                       })

    app_use = tf.decode_raw(features['app_use'], tf.float32)
    dvc = tf.cast(features['dvc'], tf.string)
    label = tf.cast(features['label'], tf.int64)
    #     app_use = tf.cast(features['app_use'], tf.float32)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        batch = sess.run([app_use])
        print(batch)
    #         print(np.array(batch).shape)

    coord.request_stop()
    coord.join(threads)
    return app_use
'''
def main():
    # Convert to Examples and write the result to TFRecords.
    convert_to('newTest.txt', 'newTest_sample_month')
    # read_and_decode('train')


if __name__ == '__main__':
    main()


# In[26]:


f = open('newTest.txt')
label = [];index = 0;l0=0; l1=0
for i in f.readlines():
    line = i.strip().split('\t')
    label.append(line[4])
    if line[4] == '0.0' :
        l0 += 1
    elif  line[4] == '1.0' :
        l1 += 1
    else:
        print(line)
print(l0)
print(l1)
print(len(label))




# In[1]:


'''
将用户的app安装列表转化为最后一天（1*7000）和一个月（31*7000）的csv数据，主要用来做keras和单机运行
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import pickle as pk
import h5py
import argparse


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_app_list():
    #读app列表，可选数量为11万和7000
    app_list=[]
    f1 = open('app_list.txt',encoding='utf-8')
    for i in f1.readlines():
        line =i.strip()
        app_list.append(line)
    f1.close()
    print(app_list[:3])
    return app_list


def gen_data(periods, n_feature, app_info,time, app_list, is_last_row):
    #json_info = str(app_info)
    last_time = time
 
    df = pd.DataFrame(np.zeros(shape=(periods, n_feature)), columns=app_list)
    for rec in eval(app_info):
        #print(type(rec))
        date_list = pd.Series(np.zeros(periods), index=pd.date_range(end=last_time, periods=periods))
        for date in rec['load_info']:
            date_list[date] = 1

        df[rec['app_name'].strip()] = date_list.values
    '''
    is_last_row=True，表示只取最后一天的安装列表
    is_last_row=False，表示取一个月的安装列表，且需要注释掉下面一行代码！！！！！！！！
    
    '''
    #df1 = df.loc[~(df == 0).all(axis=1)]
    
    if is_last_row:
        #newTest中有空值，所以下面代码会报错，加try后依然有错，data未声明，需要重新赋值
        try:
            data = df1.iloc[-1]
        except IndexError:
            #print(df)
            data = df.iloc[-1]
            #print(data)
    else:
        data = df1

    #print('gen_data is over!')
    return data


def convert_to(data_set, name):
    #file = h5py.File(data_set, 'r')
    f = open(data_set)
    dvc_list = [];time = [];app_info = [];label_list = [];aa = 0
    for i in f.readlines():
        line = i.strip().split('\t')

        #line[3] = eval(line[3])
        dvc_list.append(line[1])
        time.append(line[2])
        app_info.append(line[3])
        label_list.append(line[4])

    
    app_list = get_app_list()
    periods = 31;n_feature = len(app_list)
    print(n_feature)
    app_use = [];df = pd.DataFrame(np.zeros(shape=(periods, n_feature)), columns=app_list)
    for index,item in enumerate(app_info):
        app = np.array(gen_data(periods=31, n_feature=len(app_list),app_info=item,time=time[index], app_list=app_list, is_last_row=False))
        df.loc[index] = app
        #app_use.append(app)
        
    df['label'] = pd.DataFrame(label_list)
    filename = name + '.csv'
    df.to_csv(filename,index=0,encoding='utf-8')
    print('Done')

    
def main():
    # Convert to Examples and write the result to TFRecords.
    convert_to('newTest.txt', 'newTest_sample_last')
    # read_and_decode('train')


if __name__ == '__main__':
    main()


# In[ ]:


'''
读取tfrecord格式的代码
'''
import tensorflow as tf
import argparse
import os
import numpy as np

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
#     lsi.set_shape([features['height'], features['width'], 1])
    app_use = tf.reshape(app_use, shape=[1, 7120, 1])
    #app_use = tf.cast(app_use, tf.float32)

#     dvc = tf.cast(features['dvc'], tf.string)
    
    label = tf.decode_raw(features['label'], tf.int64)
#     label.set_shape([1, 2])

    #     label = tf.cast(label, tf.int64)
    
    return label, app_use


def read_shuffle_feature(batch_size, shuffle_buffer, epochs, name):
#     filename = os.path.join(FLAGS.directory, name + '.tfrecords')
#     filename = 'hdfs://ns-hf/user/mzzhao/lsi.tfrecords'
    filename = './test_last_nL.tfrecords'
    dataset = tf.data.TFRecordDataset(filename)         .map(parser_tfrecord)         .shuffle(shuffle_buffer)         .batch(batch_size)         .repeat(epochs)


    iterator = dataset.make_initializable_iterator()

    # for test data
    label, lsi = iterator.get_next()

    step = 0

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(),
                  tf.local_variables_initializer()))

        sess.run(iterator.initializer)
        while True:
            try:
#                 dvc_batch = sess.run([dvc])
                label_batch = sess.run(label)
                app_batch = sess.run(lsi)
#                 print('-#-@-',label_batch)
#                 print('-#-@-',app_batch[0])
#                 print('-#-@-',np.array(app_batch).shape)
#                 print('-#-@-',np.array(label_batch).shape)
#                 print('-#-@-',np.array(app_batch))
#                 print('-#-@-',np.array(label_batch))
            except tf.errors.OutOfRangeError:
                break

            step += 1
            
            if step > 0 and step % 100 == 0:
                  print(step)
                
    return iterator

batch_size = 64
shuffle_buffer = 1000
num_epochs = 5
read_shuffle_feature(batch_size, shuffle_buffer, num_epochs, 'test')


# In[ ]:


df = pd.DataFrame()

