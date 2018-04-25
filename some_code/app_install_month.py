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
    #df = df.loc[~(df == 0).all(axis=1)]

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
        app = np.array(gen_data(periods=31, n_feature=len(app_list),app_info=app_info[index],time=time[index], app_list=app_list, is_last_row=False))
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
    convert_to('train.txt', 'train_month_nL')
    # read_and_decode('train')


if __name__ == '__main__':
    main()
