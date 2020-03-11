#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim.models import KeyedVectors
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from os import listdir
from os.path import isfile, join
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys
from string import punctuation
TaggededDocument = gensim.models.doc2vec.TaggedDocument
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim import corpora, models, similarities
import numpy as np
import pandas as pd
import heapq
import csv

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from keras.layers import *
import keras
from gensim.models import Word2Vec
'''
测试用
le = preprocessing.LabelEncoder()
x = pd.Series(["paris", "paris", "tokyo", "amsterdam"])
xx = x.value_counts().index
label  = list(xx)


print(label)


le.fit(label)

print(list(le.classes_))

print(le.transform(["tokyo", "tokyo", "paris","amsterdam"]))
'''



df = pd.read_csv('D:/data_set/toutiao-text-classfication-dataset-master/toutiao_cat_data.txt/toutiao_cat_data.txt',sep='_!_',header=None,encoding='utf-8').iloc[:,2:4]
print(df.head(5))

#从停用词表读取停用词
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords


texts = []
apptypes = []
train_y = []

#    print(line[0])
#    print(line) #打印文件每一行的信息
#    content.append(line)
#print("该文件中保存的数据为:\n",content)
print(df.shape)
stopwords = stopwordslist('dropout.txt')
def title_preprocessing(title):
    app_desc2 = list(jieba.cut(title, cut_all=False))
    outstr = ''
    for word in app_desc2:
        if word not in stopwords:
            if word != '/t':
                outstr += word
                outstr += " "
    return outstr
for ind,item in df.iterrows():
    texts.append(title_preprocessing(item[3]))
    if item[2] not in apptypes:
        apptypes.append(item[2])
    train_y.append(apptypes.index(item[2]))
    if ind%10000==0:
        print('->',end='')


print(len(set(train_y)))

#print(texts)




# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(texts, train_y, test_size=0.1, random_state=42)

# 对类别变量进行编码，共48类
'''
y_labels = list(y_train)
le = preprocessing.LabelEncoder()
le.fit(y_labels)
'''

#print(y_train)
num_labels = 15
y_train = to_categorical(y_train, num_labels)
#y_test = to_categorical(y_test, num_labels)

# 分词，构建单词-id词典       
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(texts)
vocab = tokenizer.word_index
#print(vocab)
# 将每个词用词典中的数值代替
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)
'''
# One-hot
x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')
'''
# 序列模式
x_train = pad_sequences(X_train_word_ids, maxlen=20)
x_test = pad_sequences(X_test_word_ids, maxlen=20)

#print(x_test)
print(len(y_train[0]))



def CNN_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50)) #使用Embeeding层将每个词编码转换为词向量
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, y_train,epochs=5, batch_size=80)
    y_predict = model.predict_classes(x_test_padded_seqs)  # 预测的是类别，结果就是类别号
#    y_predict = list(map(str, y_predict))

 #   y_predict = to_categorical(y_predict, num_labels)
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))




#y_test = list(map(str, y_test))
#CNN_model(x_train,y_train,x_test,y_test)

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('D:/工作交接/liusq/liusq/Google_word2vec_zhwiki1709_300d.bin',binary=True)# 预训练的词向量中没有出现的词用0向量表示
embedding_matrix = np.zeros((len(vocab) + 1, 300))
for word, i in vocab.items():
    try:
        embedding_vector = w2v_model[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue
 
#构建TextCNN模型
def TextCNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embedding_matrix):
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(20,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=20, weights=[embedding_matrix], trainable=False)
    #embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=18)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=17)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=16)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.5)(flat)
    main_output = Dense(15, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
#    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, y_train, batch_size=1000, epochs=1)
    #y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    y_predict = np.argmax(result, axis=1)  # 获得最大概率对应的标签
 #   y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

    # mp = "textCNN_model_toutiao_clf.h5"
    # model.save(mp)

TextCNN_model(x_train,y_train,x_test,y_test,embedding_matrix)





def LSTM_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embedding_matrix):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(48, activation='softmax'))
#    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    model.fit(x_train_padded_seqs, y_train,epochs=20, batch_size=800) #训练时间为若干个小时
    y_predict = model.predict_classes(x_test_padded_seqs)
#    classes = model.predict_classes(xa)
#    acc = np_utils.accuracy(classes, ya)
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))


#LSTM_model(x_train,y_train,x_test,y_test,embedding_matrix)