from sklearn import preprocessing,metrics
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import plot_model
def data_generate(a,b):
    res =[]
    for i in range(a):
        for j in range(b):
            add_num = str(i+j)
            data=str(i)+'+'+str(j)
            res.append([data,add_num])
            # dot_num =str(i*j)
            # data1 = str(i)+'*'+str(j)
            # res.append([data1,dot_num])
    return res


res= data_generate(1000,1000)

df= pd.DataFrame(res,columns=['inputs','targets'])
print(df.head(5))


N_UNITS = 256
BATCH_SIZE = 64
EPOCH = 30
NUM_SAMPLES = 10000

#
# df = pd.read_csv('D:/python_project/NLP/cmn_eng/cmn.txt',sep='\t',header=None).iloc[:NUM_SAMPLES,:2]
# print(df.head(5))
# df.columns=['inputs','targets']


#讲每句中文句首加上'\t'作为起始标志，句末加上'\n'作为终止标志
df['targets'] = df['targets'].apply(lambda x: '\t'+x+'\n')

print(df.head(5))
input_texts = df.inputs.values.tolist()#英文句子列表
target_texts = df.targets.values.tolist()#中文句子列表

#确定中英文各自包含的字符。df.unique()直接取sum可将unique数组中的各个句子拼接成一个长句子
input_characters = sorted(list(set(df.inputs.unique().sum())))
target_characters = sorted(list(set(df.targets.unique().sum())))

INPUT_LENGTH = max([len(i) for i in input_texts])
OUTPUT_LENGTH = max([len(i) for i in target_texts])
INPUT_FEATURE_LENGTH = len(input_characters)
OUTPUT_FEATURE_LENGTH = len(target_characters)





#encoder输入、decoder输入输出初始化为三维向量
encoder_input = np.zeros((NUM_SAMPLES,INPUT_LENGTH,INPUT_FEATURE_LENGTH))
decoder_input = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
decoder_output = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))



input_dict = {char:index for index,char in enumerate(input_characters)}
input_dict_reverse = {index:char for index,char in enumerate(input_characters)}
target_dict = {char:index for index,char in enumerate(target_characters)}
target_dict_reverse = {index:char for index,char in enumerate(target_characters)}



#encoder的输入向量one-hot
for seq_index,seq in enumerate(input_texts):
    for char_index, char in enumerate(seq):
        encoder_input[seq_index,char_index,input_dict[char]] = 1

#decoder的输入输出向量one-hot，训练模型时decoder的输入要比输出晚一个时间步，这样才能对输出监督
for seq_index,seq in enumerate(target_texts):
    for char_index,char in enumerate(seq):
        decoder_input[seq_index,char_index,target_dict[char]] = 1.0
        if char_index > 0:
            decoder_output[seq_index,char_index-1,target_dict[char]] = 1.0


def create_model(n_input,n_output,n_units):
    #训练阶段
    encoder_input = Input(shape = (None, n_input))
    encoder = LSTM(n_units, return_state=True)
    _,encoder_h,encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h,encoder_c]

    #decoder
    decoder_input = Input(shape = (None, n_output))
    decoder = LSTM(n_units,return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder(decoder_input,initial_state=encoder_state)
    decoder_dense = Dense(n_output,activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    #生成的训练模型
    model = Model([encoder_input,decoder_input],decoder_output)

    #推理阶段，用于预测过程
    encoder_infer = Model(encoder_input,encoder_state)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]#上个时刻的状态h,c

    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]#当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)#当前时刻的输出
    decoder_infer = Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)

    return model, encoder_infer, decoder_infer


def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    state = encoder_inference.predict(source)
    #第一个字符'\t',为起始标志
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,target_dict['\t']] = 1
    output = ''
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        char = target_dict_reverse[char_index]
        output += char
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
        if char == '\n':#预测到了终止符则停下来
            break
    return output


model_train, encoder_infer, decoder_infer = create_model(INPUT_FEATURE_LENGTH, OUTPUT_FEATURE_LENGTH, N_UNITS)

# plot_model(model=model_train,show_shapes=True)


model_train.compile(optimizer='adam',loss='categorical_crossentropy')
model_train.summary()
encoder_infer.summary()
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0.2)

model_train.save('model_train.h5')
encoder_infer.save('encoder_infer.h5')
decoder_infer.save('decoder_infer.h5')

model_train.load_weights('calculate.h5')
encoder_infer =encoder_infer.load_weights('encoder_infer.h5')
decoder_infer = decoder_infer.load_weights('decoder_infer.h5')


for i in range(1090,1190):
    test = encoder_input[i:i+1,:,:]#i:i+1保持数组是三维
    out = predict_chinese(test,encoder_infer,decoder_infer,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH)
    print(input_texts[i])
    print(out)

# import matplotlib.pyplot as plt
# plt.imshow(encoder_input[1001])
# plt.show()
# plt.imshow(encoder_input[1002])
# plt.show()
print(encoder_input.shape)
y_predict=[]
for i in range(10000):
    y_predict.append(predict_chinese(encoder_input[i:i+1,:,:],encoder_infer,decoder_infer,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
print(len(y_predict))



y_predict1= [int(x) for x in y_predict]
y_true =[int(x) for x in list(df.targets)]
print(y_predict1[0:10])
print(y_true[:20])
print('准确率', metrics.accuracy_score(y_true, y_predict1))
print(y_predict[50:100])