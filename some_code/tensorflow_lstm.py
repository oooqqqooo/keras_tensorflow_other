import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BATCH_SIZE = 64
STEP_LEN = 100
CLASS_NUMBER = 2
HIDDEN_NUM = 200
from tensorflow.contrib.rnn import BasicLSTMCell
from keras.preprocessing import sequence


def get_pick():
    import pickle
    from sklearn.model_selection import train_test_split
    from keras.preprocessing import sequence
    with open('pickfile/data_weight_append_xall.txt', 'rb') as f:
        pf = pickle.load(f)
        embedding_weights, n_sam, x_data_grouped, y_data, x_data = pf
    with open('pickfile/data_timesort_without_short_sentences.txt', 'rb') as f:
        pf2 = pickle.load(f)
        x_data_grouped_timesort, y_data_timesort, _ = pf2
    with open('pickfile/data_shuffle1_without_short_sentences_.txt', 'rb') as f:
        pf3 = pickle.load(f)
        x_data_grouped_shuffle, y_data_shuffle, _ = pf3
    with open('pickfile/data_shuffle2_without_short_sentences.txt', 'rb') as f:
        pf3 = pickle.load(f)
        x_data_grouped_shuffle2, y_data_shuffle2, xaa = pf3

    def get_train_test(x_data_grouped_timesort, x_data_grouped_shuffle, x_data_grouped_shuffle2, y_data_timesort,
                       test_size=0.2):
        data_index = np.arange(0, len(y_data_timesort))
        np.random.shuffle(data_index)
        train_index = data_index[:int((1 - test_size) * len(y_data_timesort))]
        test_index = data_index[int((1 - test_size) * len(y_data_timesort)):]
        print(train_index)
        print(type(train_index))
        print(type(x_data_grouped_timesort), type(x_data_grouped_shuffle), type(x_data_grouped_shuffle2),
              type(y_data_timesort))
        # x_data_grouped_timesort = x_data_grouped_timesort.tolist()
        # x_data_grouped_shuffle = x_data_grouped_shuffle.tolist()
        # x_data_grouped_shuffle2 = x_data_grouped_shuffle2.tolist()
        y_data_timesort = y_data_timesort.tolist()

        train_index = train_index.tolist()
        test_index = test_index.tolist()
        print(train_index)
        print()
        x_train = [x_data_grouped_timesort[i] for i in train_index]
        y_train = [y_data_timesort[i] for i in train_index]
        x_test = [x_data_grouped_timesort[i] for i in test_index]
        y_test = [y_data_timesort[i] for i in test_index]
        for i in train_index:
            if y_data_timesort[i] == 1:
                x_train.append(x_data_grouped_shuffle[i])
                y_train.append(y_data_timesort[i])
                x_train.append(x_data_grouped_shuffle2[i])
                y_train.append(y_data_timesort[i])
        for i in test_index:
            if y_data_timesort[i] == 1:
                x_test.append(x_data_grouped_shuffle[i])
                y_test.append(y_data_timesort[i])
                x_test.append(x_data_grouped_shuffle2[i])
                y_test.append(y_data_timesort[i])
        oe = OneHotEncoder(2)
        # tmp = np.ones(len(y_data))
        # y_data = tmp - y_data
        y_data = np.reshape(y_data_timesort, (-1, 1))
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
        oe.fit(y_data)
        y_train = oe.transform(y_train).toarray()
        y_test = oe.transform(y_test).toarray()
        return x_train, x_test, y_train, y_test

    X_train, X_test, y_train, y_test = get_train_test(x_data_grouped_timesort, x_data_grouped_shuffle,
                                                      x_data_grouped_shuffle2, y_data_timesort)
    X_train = sequence.pad_sequences(X_train,
                                     maxlen=100, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=100, padding='post')
    return embedding_weights, n_sam, X_train, X_test, y_train, y_test




def myRNN(x, weights, biases, lengths):
    embbbb = tf.constant(value=embedding_weights, dtype=tf.float32)
    x = tf.nn.embedding_lookup(embbbb, x)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_NUM)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, sequence_length=lengths,
                                        dtype=tf.float32)
    output = tf.matmul(outputs[:, -1, :], weights['out']) + biases['out']
    return output


def get_auc(y_true, y_pred):
    au = metrics.roc_auc_score(y_true, y_pred, average='macro')
    return au


if __name__ == '__main__':
    embedding_weights, n_sam, X_train, X_test, y_train, y_test = get_pick()

    length_train = [len(i) for i in X_train]
    print(length_train)
    length_test = [len(i) for i in X_test]
    # X_train = sequence.pad_sequences(X_train, maxlen=100, padding='post')
    # X_test = sequence.pad_sequences(X_test, maxlen=100, padding='post')
    x_placeholder = tf.placeholder(name='x_placeholder', shape=[None, STEP_LEN], dtype=tf.int64)
    y_placeholder = tf.placeholder(name='y_placeholder', shape=[None, 2], dtype=tf.float32)
    l_placeholder = tf.placeholder(name='l_placeholder', shape=[None], dtype=tf.int32)
    loss_placeholder = tf.placeholder(name='loss_placeholder',shape=[None],dtype=tf.float32)
    weights = {
        'out': tf.Variable(name='weight', initial_value=tf.random_normal([HIDDEN_NUM, CLASS_NUMBER]))
    }
    biases = {
        'out': tf.Variable(name='biases', initial_value=tf.random_normal([CLASS_NUMBER]))
    }
    pred = myRNN(x_placeholder, weights, biases, l_placeholder)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=pred), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('graph/graph1', sess.graph)
        batch_start = 0

        for _ in range(10000):
            batch_x = X_train[batch_start:batch_start + BATCH_SIZE]
            batch_y = y_train[batch_start:batch_start + BATCH_SIZE]
            batch_l = length_train[batch_start:batch_start + BATCH_SIZE]
            # batch_x_test = X_test[batch_start:batch_start + BATCH_SIZE]
            # batch_y_test = y_test[batch_start:batch_start + BATCH_SIZE]
            # batch_l_test = length_test[batch_start:batch_start + BATCH_SIZE]
            if batch_start + BATCH_SIZE > len(y_train):
                batch_start = 0
            else:
                batch_start += BATCH_SIZE
            o, acc, los = sess.run([optimizer, accuracy, loss],
                                   feed_dict={x_placeholder: batch_x, y_placeholder: batch_y, l_placeholder: batch_l})
            if _ % 50 == 0:
                val_acc = sess.run(accuracy, feed_dict={x_placeholder: X_test, y_placeholder: y_test,
                                                        l_placeholder: length_test})
                val_los = sess.run(loss, feed_dict={x_placeholder: X_test, y_placeholder: y_test,
                                                    l_placeholder: length_test})
                val_pre = sess.run(pred, feed_dict={x_placeholder: X_test, y_placeholder: y_test,
                                                    l_placeholder: length_test})

                auc = get_auc(y_test, val_pre)
                print(_, 'loss-->', los, '  acc-->', acc, '   val_los-->', val_los, '   val_acc-->', val_acc, ' val_auc-->',
                      auc)
        val_pre = sess.run(pred, feed_dict={x_placeholder: X_test, y_placeholder: y_test,
                                            l_placeholder: length_test})

        writer.close()
