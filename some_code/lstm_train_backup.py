import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

STEP_LEN = 100
VEC_LEN = 200


def preprocessing():
    '''

    :return: a df merged by words after grouped and label
    '''
    df_words = pd.read_csv('data/md5_ID_time.txt', sep='	')
    df_words = df_words.rename(
        columns={'2017-01-01 20:42:47': 'times', 'e40a2603a012a6a9e3c12ff518f390d9': 'dev',
                 '[4421, 1175]': 'sentences'})
    # df_words.sort_values()
    df_labels = pd.read_csv('data/label_use.csv')
    df_labels = df_labels.rename(columns={'4aa4d608bbea479034b1203862e46bf8': 'dev', '0': 'label'})

    df_words = df_words.groupby(['dev']).aggregate(lambda x: list(x)).reset_index()
    print('df word', len(df_words))
    df = pd.merge(df_words, df_labels, how='inner')
    return df


def get_embedding():
    import numpy as np
    with open('data/id_embedding.txt') as f:
        lines = f.readlines()
        n_sam = len(lines)
        embedding_weights = np.zeros((n_sam, VEC_LEN))
        import re
        pattern = re.compile("(\d*)(.*)")

        for line in lines:
            matc = pattern.match(line)
            index = matc.group(1)
            vec = matc.group(2)
            try:
                index = int(index)
                vec = eval(vec)
                vec = [float(i) for i in vec]
                embedding_weights[index, :] = vec
            except Exception as e:
                continue

    return embedding_weights, n_sam
    # print(vec.__len__())
    # print(vec)


def data_grouped(df):
    import itertools
    from keras.preprocessing import sequence
    x_data = df['sentences']
    data = df[['sentences', 'label']]
    y_data = df['label']
    x_data_grouped = []
    y_data_new = []
    num = 0
    for row in data.index:
        sentence = data.loc[row].values[0]
        sentence = [eval(i) for i in sentence]
        label = data.loc[row].values[1]
        sentence = list(itertools.chain(*sentence))
        x_data_grouped.append(sentence)
        y_data_new.append(label)
        num += 1
    # for sentence,label in x_data,y_data:
    #     sentence = [eval(i) for i in sentence]
    #     y_data_new =[eval(i) for i in label]
    #     sentence = list(itertools.chain(*sentence))
    #     x_data_grouped.append(sentence)
    #     y_data_new.append()
    #     num += 1
    # /////跑代码要删
    # if num > 100:
    #     break
    x_data_grouped = sequence.pad_sequences(x_data_grouped, maxlen=STEP_LEN)
    y_data_new = np.array(y_data_new)
    print(y_data_new)
    print(y_data_new.__len__())
    print(y_data_new.shape)
    print(type(y_data_new))
    return x_data_grouped, y_data_new


def get_train_test(X, y, test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def metrics_result(actual, predict):
    from sklearn import metrics
    acc = metrics.accuracy_score(actual, predict)
    print(acc)
    prediction = metrics.precision_score(actual, predict, average=None)

    print(prediction)
    recall = metrics.recall_score(actual, predict, average=None)

    f1 = metrics.f1_score(actual, predict, average=None)
    return acc, prediction, recall, f1


def train_lstm(embedding_weights, n_sam, X_train, X_test, y_train, y_test):
    from keras.models import Sequential
    from keras.layers.recurrent import LSTM
    from keras.layers.core import Dense, Masking, Activation, Dropout
    from keras.layers.embeddings import Embedding
    import itertools
    model = Sequential()
    print(embedding_weights[:10])
    print(embedding_weights.shape)
    model.add(Embedding(input_dim=n_sam, output_dim=VEC_LEN, weights=[embedding_weights], input_length=STEP_LEN))
    model.add(Masking(mask_value=0))
    model.add(
        LSTM(units=200, activation='sigmoid', recurrent_activation='hard_sigmoid', recurrent_dropout=0.4, dropout=0.4))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation(activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
    predict = model.predict_classes(X_test)
    predict = np.array(list(itertools.chain(*predict)))
    print(type(y_test))
    print('sum------------------------------------------', sum(y_data))
    print(type(predict))
    print(predict)

    acc, prediction, recall, f1 = metrics_result(y_test, predict)
    print('acc', acc)
    print('prediction', prediction)
    print('recall', recall)
    print('f1', f1)


#

if __name__ == '__main__':
    df = preprocessing()
    # y_data = df['label']
    x_data_grouped, y_data = data_grouped(df)
    print(x_data_grouped.shape)
    print(y_data)
    print('xdatalen', x_data_grouped.__len__())
    print('ydatalen', len(y_data))
    embedding_weights, n_sam = get_embedding()
    X_train, X_test, y_train, y_test = get_train_test(x_data_grouped, y_data, 0.2)
    '''
    import pickle
    with open('pickfile/data_weight.txt','wb') as f:
        pickle.dump([embedding_weights, n_sam, X_train, X_test, y_train, y_test],f)'''
    train_lstm(embedding_weights, n_sam, X_train, X_test, y_train, y_test)
