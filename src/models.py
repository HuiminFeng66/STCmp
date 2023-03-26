"""
General comments:
1 Note that LFZip(NLMS),LFZip(NN) and CA(critical aperture) expect the input to be in numpy array(.npy) format and support only float32 arrays
2 LFZip(NLMS) additionally supports multivariable time series with at most 256 variables,where the input is a numpy array of shape(k,T),
where k is the number of variables and T is the length of the time series.
3 During compression,the reconstructed time series is also generated as a byproduct and stored as compressed_file.bsc.recon.npy.
This can be used to verify the correctness of the compression-decompression pipeline
4 Exampples are shown after the usage below
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dropout
# from matplotlib import pyplot
# from tensorflow import keras
import keras

##
# from tensorflow.contrib.seq2seq.python.ops import *
import tensorflow as tf
# from /src/tpa-relation/AttLayer import AttentionLayer

import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Reshape,Lambda,Flatten,Conv2D
from tensorflow.keras.layers import Layer,Dense,LSTM,Activation,Multiply,Add,concatenate
from ConvAttLayer1 import CalculateScoreMatrix
##########
from attentionlayer1 import AttentionLayer
from attentionlayer2 import Attention
from DARNNLayer import Encoder,Decoder
import numpy as np
from SENet import Squeeze_excitation_layer,Squeeze_excitation_layer1
import math
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from PIL import Image
from AR1 import AR


# models for floating point data_set compression
# new models can be added as functions here
# note that the first parameter to any function should be the input_dim


"""
# 建立全连接神经网络FC
def FC(input_dim,num_hidden_layers,hidden_layer_size):    # 输入维度、隐藏层数、每层隐藏层的大小（神经单元个数）
    if num_hidden_layers == 0:
        model = Sequential()
        model.add(Dense(1,input_dim=input_dim))   # 1表示输出空间维度
        return model
    assert num_hidden_layers > 0
    model = Sequential()
    # 最简单的序贯模型、序贯模型是多个网络层的线性堆叠
    model.add(Dense(hidden_layer_size, activation='relu',input_dim=input_dim))
    # Dense为全连接层
    # 第一层隐藏层为全连接层，有hidden_layer_size个神经单元
    model.add(BatchNormalization())
    # Batch Normalization（BN）是深度学习中非常好的算法，加入BN层的网络会更加稳定并且BN还起到了一定正则化的作用
    for i in range(num_hidden_layers-1):
        model.add(Dense(hidden_layer_size, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dense(1))     # ? the prediction value?
    # 输出层、只有一个神经单元
    return model
    # (single) [32,4,128] val_loss did not improve from 0.32739
    # (single) compressed 788480 into 338186 in 0.206 seconds


def biGRU(input_dim, num_biGRU_layers, num_units_biGRU, num_units_dense):     # num_units_dense?
        assert num_biGRU_layers > 0
        model = Sequential()
        model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))   # input_shape=(input_dim,)  1D data,input_dim=32; reshape(input_dim,1) 2D data,input_dim=2
        # keras.layers.Reshape实现keras不同层的对接
        # 但注意的是，它毕竟改变了数据的维度，也就改变了数据的组成方式，这可能会对后续的识别产生影响
        for i in range(num_biGRU_layers-1):
            model.add(Bidirectional(GRU(num_units_biGRU, return_sequences=True)))    # return_sequence=True:output all time step's GRU hidden state
            # 是否返回最后一个输出或是整个序列的输出，默认是False
            # Bidirectional 双向GRU
        model.add(Bidirectional(GRU(num_units_biGRU, return_sequences=False)))      # return_sequence=False:only output the latest GRU hidden state
        model.add(Dense(num_units_dense, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model
"""

def mine(input_dim,time_steps):
    output_dim=input_dim
    filters=36  # can sqrt
    units=32  ####
    inp=tf.keras.Input(shape=(time_steps,input_dim))
    print('inp.shape:', inp.shape)       # inp.shape(?, 16, 200)
    inp1=tf.transpose(inp,[0,2,1])
    print('inp1.shape:',inp1.shape)      # inp1.shape(?, 200, 16)
    # conduct conv on inpto extract temporal pattern
    HC = Conv1D(filters=filters, kernel_size=1, strides=1, padding="causal")(inp1)   # shape=(input_dims,filters)
    print('math.sqrt(filters):', math.sqrt(filters))  # math.sqrt(time_steps): 6.0
    # filters must can be sqrt
    s = int(math.sqrt(filters))
    # how to reshape?    random ??   the question is here
    CHW = Reshape((input_dim, s, s))(HC)
    print('CHW.shape:',CHW.shape)   # H_CHW.shape: (?, 200, 6, 6)
    HWC = tf.transpose(CHW, [0, 2, 3, 1])
    print('HWC.shape:', HWC.shape)   # H_HWC.shape: (?, 6, 6, 200)
    H3 = HWC
    print('H3.shape:', H3.shape)    # H3.shape: (?, 6, 6, 200)
    # use SENet
    excitation = Squeeze_excitation_layer1(input_x=H3, out_dim=input_dim, ratio=4, layer_name='squeeze_exc_layer')
    print(excitation.shape)  # (?,200,1)
    # Vt=H1*excitation    # wrong
    # print('Vt.shape:',Vt.shape)  # Vt.shape: (?, 32, 16)
    Vt = Multiply()([inp1, excitation])
    print('Vt.shape:', Vt.shape)  # Vt.shape: (?, 200, 16)
    V = Lambda(lambda x: K.sum(x, axis=1))(Vt)
    print('V.shape:', V.shape)  # V.shape: (?, 16)
    out = Dense(output_dim, activation="sigmoid", name='2dense')(V)
    print('out.shape:', out.shape)  # out.shape: (?, 200)

    # add AR
    ar=AR(input_dim,time_steps)(inp)
    print('ar.shape:',ar.shape)   # ar.shape: (?, 200)
    final_out=Add()([out,ar])
    print('final_out.shape:',final_out)   # final_out.shape: Tensor("add/add:0", shape=(?, 200), dtype=float32)
    model = Model(inputs=inp, outputs=final_out)
    model.summary()
    return model




def TPA(input_dim ,time_steps, num_units_LSTM):   #    TPA model,focus on temporal attention weights,ignoring spatio attention weights.
    # self.epochs = epochs
    # self.feat_dim = None
    # feat_dim=input_dim
    # self.input_dim = None
    # input_dim=time_steps
    # self.output_dim=None
    output_dim = input_dim   #
    # self.lr = learning_rate
    filters = 32               # num_units_LSTM
    # self.batch_size = batch_size
    units=num_units_LSTM       ##  units=num_units_LSTM
    # self.model_name = "TPA-LSTM"
    filter_size=3
    # batch_size=32
    inp=tf.keras.Input(shape=(time_steps,input_dim))
    print('inp.shape:',inp.shape)  # inp.shape: (?, 30, 250)
    # convolution layer
    # x=Conv1D(filters=filters,kernel_size=filter_size,strides=1,padding="causal")(inp)
    # print('x.shape:',x.shape)    # x.shape: (?, 32, 32)
    # LSTM layer
    x=LSTM(units=units,
           # kernel_initializer="glorot_uniform",
           kernel_initializer="random_uniform",
           bias_initializer="zeros",
           kernel_regularizer=regularizers.l2(0.001),
           return_sequences=True)(inp)
    # get the 1~t-1 and t hidden state
    H=Lambda(lambda x:x[:,:-1,:])(x)        # H.shape: (?, 29, 32)
    print('H.shape:',H.shape)
    ht=Lambda(lambda x:x[:,-1,:])(x)
    ht=Reshape((units,1))(ht)
    print('ht.shape',ht.shape)              # ht.shape (?, 32, 1)

    # get the HC by 1*1 convolution
    H = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)   # from H(t,m) reshape to H(m,t) for conv
    print('H.shape:',H.shape)             # H.shape: (?, 32, 29)
    HC=Conv1D(filters=filters,kernel_size=1,strides=1,padding="causal")(H)
    print('HC2.shape:',HC.shape)      #  (?, 32, 32)
    score_mat = CalculateScoreMatrix(units)(HC)     #   goal is to make the dimension be equal
    score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])  # get the score
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat)
    print('HC.shape:',HC.shape)  # HC.shape: (?, 32, 32)
    print('score_mat.shape:',score_mat.shape)  # score_mat.shape: (?, 32, 1)
    attn_mat = Multiply()([HC, score_mat])
    attn_vec = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat)
    # get the final prediction
    wvt = Dense(units=filters , activation=None)(attn_vec)    ##########filters*4
    print('wvt.shape:',wvt.shape)
    wht = Dense(units=filters , activation=None)(Flatten()(ht))
    print('wht.shape:',wht.shape)
    yht=Add()([wht, wvt])             ## yht=Multiply()([wht, wvt])
    print('yht.shape:',yht.shape)
    # yht = Add()([wht, wvt])   ######################
    # get the output
    out = Dense(output_dim, activation="sigmoid")(yht)
    print('out.shape:',out.shape)         # out.shape: (?, 250)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model






def SEnet(input_dim,time_steps,num_units_LSTM):   # (3,16,16)    the original version
    output_dim = input_dim
    units = num_units_LSTM           ##  units = num_units_LSTM *2
    filters=32
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:', inp.shape)    # inp.shape: (?, 16, 3)
    print(inp)
    # spatio correlation, transform 1D into 2D,and padding with zero
    ##############################################
    # m=math.ceil(math.sqrt(input_dim))
    # inp0=Reshape([time_steps,m,m])(inp)    # reshape first,,  the question is how to padding with zero
    # print(inp0)
    # print('1inp0.shape:',inp0.shape)   # (?, 16, 2, 2)
    # # inp=tf.pad(inp,[[0,0],[1,1],[1,1]],constant_values=0)
    # # print('2inp.shape:',inp.shape)
    # inp00=Reshape([m,m,time_steps])(inp0)

    #####################################################
    # m=math.ceil(math.sqrt(input_dim))
    # print('m:',m)
    # n=int(math.pow(m,2)-input_dim)   # n represents the number of zeros to be padded
    # print('n:',n)
    # inp0=tf.pad(inp,[[0,0],[0,0],[0,n]],constant_values=0)    # padding n zeros in dim=2
    # print('inp.shape:',inp0.shape)    # inp has been padded
    # # to reshape inp from 2D to 3D
    # inp00=Reshape([time_steps,m,m])(inp0)
    # print(inp00)
    # print('inp00.shape:', inp00.shape)        # inp00.shape: (?, 16, 2, 2)
    # inp000 = Reshape([m, m, time_steps])(inp00)

    #####################################################
    # inp5=Conv2D(filters=time_steps,kernel_size=[2,m],strides=[1,1],padding="same",data_format='channels_last')(inp000)
    # print('ok')
    # print('inp5.shape',inp5.shape)
    # # inp=Reshape([-1,-1,inp[3],inp[2]])(inp)
    # # print('3inp.shape',inp.shape)
    # print('inp5.shape[1]',inp5.shape[1])
    # print('inp5.shape[2]',inp5.shape[2])
    # inp6=Reshape([time_steps,inp5.shape[1]*inp5.shape[2]])(inp5)
    # print('inp6.shape:',inp6.shape)
    # # # input data into LSTM
    # # temporal correlation
    x = LSTM(units=units,        # units=num_units_LSTM
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(inp)
    # get the 1~t-1 and t hidden state
    H = Lambda(lambda x: x[:, :, :])(x)
    print('H1.shape:', H.shape)        # H1.shape: (?, 16, 32)
    ht = Lambda(lambda x: x[:, -1, :])(x)
    ht = Reshape((units, 1))(ht)
    print('ht.shape', ht.shape)  # ht.shape (?, 32, 1)
    ####################################
    # reshape 2D H(t,m) into 2D H(m,t),m represents dimension
    H = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
    print('H2.shape:',H.shape)              # H2.shape: (?, 32, 16)
    # reshape 2D H into 3D H, so that can operate SENet, channels C is dimension m
    print('math.sqrt(time_steps):',math.sqrt(time_steps))    # math.sqrt(time_steps): 4.0
    #########################################
    # time_steps must can be sqrt
    s=int(math.sqrt(time_steps))
    if math.floor(math.sqrt(time_steps)+0.5)==math.sqrt(time_steps):     # is Z or not
        H=Reshape((s,s,units))(H)
        print('ok')
    else:
        H=Reshape(math.ceil(math.sqrt(time_steps)),math.ceil(math.sqrt(time_steps)),units)(H)   # channels C=units=dimensions,H=W=math.sqrt(time_steps)
        print('no')
        # H=tf.pad(H,)
    print('H3.shape:',H.shape)        # H3.shape: (?, 4, 4, 32)
    ############################################################
    # # solve the problem:the number of time_steps can be any
    # s=math.ceil(math.sqrt(time_steps))
    # t=int(math.pow(s,2)-time_steps)
    # H1=tf.pad(H,[[0,0],[0,0],[0,t]],constant_values=0)
    # print('H1.shape:',H1.shape)
    # H2=Reshape((s,s,units))(H1)
    # print('H.shape:',H2.shape)

    #############################################################
    # use SENet
    H_se=Squeeze_excitation_layer(input_x=H,out_dim=units,ratio=4,layer_name='squeeze_exc_layer')
    print('H_se.shape:',H_se.shape)         # H_se.shape: (?,4,4,32)
    H_after_se=Reshape((units,int(math.pow(s,2))))(H_se)    # from 3D to 2D          Reshape(units,time_steps)
    print('H_after_se.shape:',H_after_se.shape)        # H_after_se.shape: (?, 32,16)
    Vt= Lambda(lambda x: K.sum(x, axis=-1))(H_after_se)       # from H_se to H_after_se,just from array3D to 2D
    print('Vt.shape:',Vt.shape)         # Vt.shape: (?, 32)

    # wht = Dense(units=units, activation=None, name='1dense')(Flatten()(ht))
    # print('wht.shape:', wht.shape)  # wht.shape: (?, 32)
    # yht = Multiply()([wht, Vt])
    # print('yht.shape:', yht.shape)  # yht.shape: (?, 32)
    out = Dense(output_dim, activation="sigmoid", name='2dense')(Vt)
    print('out.shape:', out.shape)  # out.shape: (?, 200)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model






def SEnet1(input_dim,time_steps,num_units_LSTM):   # (200,16,32)             modify on TPA,change CHW to HWC
    output_dim = input_dim
    units = num_units_LSTM           ##  units = num_units_LSTM *2
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:', inp.shape)    # inp.shape: (?, 16, 200)
    print(inp)
    # # input data into LSTM
    # # temporal correlation
    x = LSTM(units=units,                  # units=num_units_LSTM
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(inp)  # x.shape: (?, 16, 32)
    # get the 1~t and t hidden state
    H = Lambda(lambda x: x[:, :, :])(x)
    print('H1.shape:', H.shape)         # H1.shape: (?, 16, 32)
    ht = Lambda(lambda x: x[:, -1, :])(x)   # (?,1,32)
    ht = Reshape((units, 1))(ht)     # from row to column
    print('ht.shape', ht.shape)  # ht.shape (?, 32, 1)
    ####################################
    # reshape 2D H(t,m) into 2D H(m,t),m represents dimension
    # H = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
    H1=tf.transpose(H,[0,2,1])   # the order of elements have changed
    print('H2.shape:',H1.shape)              # H2.shape: (?, 32, 16)
    # reshape 2D H into 3D H, so that can operate SENet, channels C is dimension m
    # H_1=Reshape((units,1,time_steps))(H)
    # print('H_1.shape:',H_1.shape)       # H_1.shape  (?,32,1,16)
    print('math.sqrt(time_steps):',math.sqrt(time_steps))    # math.sqrt(time_steps): 4.0
    #########################################
    # time_steps must can be sqrt
    s=int(math.sqrt(time_steps))
    ####################################
    # how to reshape?    random ??   the question is here
    H_CHW=Reshape((units,s,s))(H1)      # (32,4,4) transform into (4,4,32)   strat from row or column
    #####################################################    (32,4,4) transform into (4,4,32)
    print('H_CHW.shape:',H_CHW.shape)
    H_HWC=tf.transpose(H_CHW,[0,2,3,1])
    print('H_HWC.shape:',H_HWC.shape)
    H3=H_HWC
    print('H3.shape:',H3.shape)        # H3.shape: (?, 4, 4, 32)
    # use SENet
    excitation=Squeeze_excitation_layer1(input_x=H3,out_dim=units,ratio=4,layer_name='squeeze_exc_layer')
    print(excitation.shape)   # (?,32,1)
    Vt=H1*excitation    # wrong
    print('Vt.shape:',Vt.shape)  # Vt.shape: (?, 32, 16)
    Vt = Lambda(lambda x: K.sum(x, axis=1))(Vt)
    print('Vt.shape:', Vt.shape)  # Vt.shape: (?, 16)
    # wht = Dense(units=units, activation=None, name='1dense')(Flatten()(ht))
    # print('wht.shape:', wht.shape)  # wht.shape: (?, 32)
    # yht = Multiply()([wht, Vt])
    # print('yht.shape:', yht.shape)  # yht.shape: (?, 32)
    out = Dense(output_dim, activation="sigmoid", name='2dense')(Vt)
    print('out.shape:', out.shape)  # out.shape: (?, 3)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model



def SEnet2(input_dim,time_steps,num_units_LSTM):   # (200,16,32)          modify on SEnet,add conv
    output_dim = input_dim
    units = num_units_LSTM           ##  units = num_units_LSTM *2
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:', inp.shape)    # inp.shape: (?, 16, 200)
    print(inp)
    # # input data into LSTM
    # # temporal correlation
    x = LSTM(units=units,                  # units=num_units_LSTM
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(inp)  # x.shape: (?, 16, 32)
    # get the 1~t and t hidden state
    H = Lambda(lambda x: x[:, :, :])(x)
    print('H1.shape:', H.shape)         # H1.shape: (?, 16, 32)
    ht = Lambda(lambda x: x[:, -1, :])(x)   # (?,1,32)
    ht = Reshape((units, 1))(ht)     # from row to column
    print('ht.shape', ht.shape)  # ht.shape (?, 32, 1)
    ####################################
    # reshape 2D H(t,m) into 2D H(m,t),m represents dimension
    # H = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
    H1=tf.transpose(H,[0,2,1])   # the order of elements have changed
    print('H2.shape:',H1.shape)              # H2.shape: (?, 32, 16)
    HC = Conv1D(filters=time_steps, kernel_size=1, strides=1, padding="causal")(H1)
    print('HC.shape:',HC.shape)             # (?,32,16)    32 rows and 16 filters
    # reshape 2D H into 3D H, so that can operate SENet, channels C is dimension m
    # H_1=Reshape((units,1,time_steps))(H)
    # print('H_1.shape:',H_1.shape)       # H_1.shape  (?,32,1,16)
    print('math.sqrt(time_steps):',math.sqrt(time_steps))    # math.sqrt(time_steps): 4.0
    #########################################
    # time_steps must can be sqrt
    s=int(math.sqrt(time_steps))
    ####################################
    # how to reshape?    random ??   the question is here
    H_CHW=Reshape((units,s,s))(HC)      # (32,4,4) transform into (4,4,32)   strat from row or column
    #####################################################    (32,4,4) transform into (4,4,32)
    print('H_CHW.shape:',H_CHW.shape)
    H_HWC=tf.transpose(H_CHW,[0,2,3,1])
    print('H_HWC.shape:',H_HWC.shape)
    H3=H_HWC
    print('H3.shape:',H3.shape)        # H3.shape: (?, 4, 4, 32)
    # use SENet
    excitation=Squeeze_excitation_layer1(input_x=H3,out_dim=units,ratio=4,layer_name='squeeze_exc_layer')
    print(excitation.shape)   # (?,32,1)
    # Vt=H1*excitation    # wrong
    # print('Vt.shape:',Vt.shape)  # Vt.shape: (?, 32, 16)

    Vt = Multiply()([H1, excitation])
    print('Vt.shape:',Vt.shape)   # Vt.shape: (?, 32, 16)
    V = Lambda(lambda x: K.sum(x, axis=1))(Vt)
    print('V.shape:', V.shape)  # V.shape: (?, 16)
    # wht = Dense(units=units, activation=None, name='1dense')(Flatten()(ht))
    # print('wht.shape:', wht.shape)  # wht.shape: (?, 32)
    # yht = Multiply()([wht, Vt])
    # print('yht.shape:', yht.shape)  # yht.shape: (?, 32)
    out = Dense(output_dim, activation="sigmoid", name='2dense')(V)
    print('out.shape:', out.shape)  # out.shape: (?, 3)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model







def LSTNet_skip(input_dim,time_steps,num_units_LSTM):
    p=time_steps  # the length of wimdow
    m=input_dim
    hidR=num_units_LSTM   # the output_dim of LSTM
    hidC=num_units_LSTM         # the output_dim of CNN
    hidS=int(num_units_LSTM/2)         # the output_dim of layer Skip-Lstm
    Ck=2         # the size of kernel of layer CNN
    skip=7        # the length of skip
    pt=int((p-Ck)/skip)
    hw=7            # the output of highway channels
    dropout=0.2
    output='no'
    # lr=0.001
    # loss=
    # clip=

    x=tf.keras.Input(shape=(time_steps, input_dim))
    print('x.shape:',x.shape)
    # CNN,without causal-dilation
    c=Conv1D(filters=hidC,kernel_size=Ck,activation='relu')(x)
    print('c.shape:',c.shape)
    c=Dropout(dropout)(c)
    print('c.shape after dropout:',c.shape)

    # RNN,common RNN
    r=GRU(hidR)(c)
    print('1r.shape:',r.shape)
    r=Lambda(lambda k:K.reshape(k,(-1,hidR)))(r)
    print('2r.shape:',r.shape)
    r=Dropout(dropout)(r)
    print('3r.shape:',r.shape)
    print('r:',r)

    # skip-RNN
    if skip>0:
        # c:batch_size*steps*filters,steps=p-Ck
        s=Lambda(lambda k: k[:,int(-pt*skip):,:])(c)
        s=Lambda(lambda k: K.reshape(k,(-1,pt,skip,hidC)))(s)
        s=Lambda(lambda k:K.permute_dimensions(k,(0,2,1,3)))(s)
        s=Lambda(lambda k:K.reshape(k,(-1,pt,hidC)))(s)

        s=GRU(hidS)(s)
        s=Lambda(lambda k: K.reshape(k,(-1,skip * hidS)))(s)
        s=Dropout(dropout)(s)
        print('s:',s)
        r=concatenate([r,s])    # concatenate
    res=Dense(m)(r)

    # highway, linear model:AR
    if hw>0:
        z=Lambda(lambda k:k[:,-hw:,:])(x)
        z=Lambda(lambda k: K.permute_dimensions(k,(0,2,1)))(z)
        # hw is setted by 7, use Dense to predict
        z=Lambda(lambda k:K.reshape(k,(-1,hw)))(z)
        z=Dense(1)(z)
        z=Lambda(lambda k:K.reshape(k,(-1,m)))(z)
        res=Add()([res,z])
        print('ok')
    if output !='no':
        res=Activation(output)(res)
        print('no')

    model=Model(inputs=x,outputs=res)
    model.summary()
    # plot_model(model,to_file="LSTNet_model.png",show_shapes=True)
    return model



def LSTNet_attn(input_dim,time_steps,num_units_LSTM):
    p = time_steps  # the length of wimdow
    m = input_dim
    hidR = num_units_LSTM  # the output_dim of LSTM
    hidC = num_units_LSTM  # the output_dim of CNN
    hidS = int(num_units_LSTM / 2)  # the output_dim of layer Skip-Lstm
    Ck = 2  # the size of kernel of layer CNN               # kernel_size has two dims, the first dim ca be defined, but the second dim is equal to input_dim.
    skip = 7  # the length of skip
    pt = int((p - Ck) / skip)
    hw = 7  # the output of highway channels
    dropout = 0.2
    output = 'no'
    # lr=0.001
    # loss=
    # clip=

    x = tf.keras.Input(shape=(time_steps, input_dim))
    print('x.shape:',x.shape)     # x.shape: (?, 16, 8)
    # CNN,without causal-dilation
    c = Conv1D(filters=hidC, kernel_size=Ck, activation='relu')(x)       # actually kernel_size=[Ck,input_dim]
    print('c.shape:',c.shape)    # c.shape: (?, 15, 16)
    c = Dropout(dropout)(c)
    print('c.shape after dropout:',c.shape)    # c.shape after dropout: (?, 15, 16)
    # RNN,common RNN
    r = GRU(hidR,return_sequences=True)(c)
    # add attention#
    r = Attention(32)(r)        # units need to be changed
    print('r:',r)
    res = Dense(m)(r)

    # highway, linear model:AR
    if hw > 0:
        z = Lambda(lambda k: k[:, -hw:, :])(x)
        z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
        # hw is setted by 7, use Dense to predict
        z = Lambda(lambda k: K.reshape(k, (-1, hw)))(z)
        z = Dense(1)(z)
        z = Lambda(lambda k: K.reshape(k, (-1, m)))(z)
        res = Add()([res, z])
        print('ok')
    if output != 'no':
        res = Activation(output)(res)
        print('no')

    model = Model(inputs=x, outputs=res)
    model.summary()
    # plot_model(model,to_file="LSTNet_model.png",show_shapes=True)
    return model




def AR(input_dim,time_steps):
    m=input_dim
    hw = 7  # the output of highway channels
    output='no'
    x = tf.keras.Input(shape=(time_steps, input_dim))
    z = Lambda(lambda k: k[:, -hw:, :])(x)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    # hw is setted by 7, use Dense to predict
    z = Lambda(lambda k: K.reshape(k, (-1, hw)))(z)
    z = Dense(1)(z)
    z = Lambda(lambda k: K.reshape(k, (-1, m)))(z)
    if output != 'no':
        z = Activation(output)(z)
        print('no')

    model = Model(inputs=x, outputs=z)
    model.summary()
    return model




def DARNN(input_dim,time_steps,num_units_LSTM):
    output_dim = input_dim
    units = num_units_LSTM     #  units = num_units_LSTM * 2
    # filter_size =
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    # convolution layer
    # X = Conv1D(filters=filters, kernel_size=filter_size, strides=1, padding="causal")(inp)
    # print('X.shape:',X.shape)       # (?, 32, 512)
    # DA-LSTM Layer
    encoder = Encoder(time_steps, num_units_LSTM)
    X_encoded = encoder(inp)  # generates the new input X_tilde for encoder
    # print('X_encoded.shape[1]:',X_encoded.shape[1])
    x = LSTM(units=units,
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(X_encoded)
    # add attention#
    r = Attention(32)(x)  # units need to be changed
    print('r:', r)
    # get the output
    out = Dense(output_dim, activation="sigmoid")(r)
    print('out.shape:', out.shape)  # the dimension should be the number of variables
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model



def Opt_DARNN(input_dim,time_steps,num_units_LSTM):
    output_dim = input_dim
    units = num_units_LSTM   # units = num_units_LSTM * 2
    # filter_size =
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:',inp.shape)
    print('inp:',inp)
    ####################################################
    # # preprocess the input time series
    m = math.ceil(math.sqrt(input_dim))
    print('m:', m)
    n = int(math.pow(m, 2) - input_dim)  # n represents the number of zeros to be padded
    print('n:', n)
    inp0 = tf.pad(inp, [[0, 0], [0, 0], [0, n]], constant_values=0)  # padding n zeros in dim=2
    print('inp.shape:', inp0.shape)  # inp has been padded
    # to reshape inp from 2D to 3D
    inp00 = Reshape([time_steps, m, m])(inp0)
    print(inp00)
    print('inp00.shape:', inp00.shape)  # inp00.shape: (?, 16, 2, 2)
    # inp000=tf.pad(inp00,[[0,0],[1,1],[1,1]],constant_values=0)
    # print('inp000.shape:',inp000.shape)
    inp000 = Reshape([m, m, time_steps])(inp00)

    ###############################################
    inp5 = Conv2D(filters=time_steps, kernel_size=[2, m], strides=[1, 1], padding="same", data_format='channels_last')(inp000)
    print('ok')
    print('inp5.shape', inp5.shape)
    # inp=Reshape([-1,-1,inp[3],inp[2]])(inp)
    # print('3inp.shape',inp.shape)
    print('inp5.shape[1]', inp5.shape[1])
    print('inp5.shape[2]', inp5.shape[2])
    inp6 = Reshape([time_steps, inp5.shape[1] * inp5.shape[2]])(inp5)
    print('inp6.shape:', inp6.shape)

    # DA-LSTM Layer
    encoder = Encoder(time_steps, num_units_LSTM)
    X_encoded = encoder(inp6)  # generates the new input X_tilde for encoder
    # print('X_encoded.shape[1]:',X_encoded.shape[1])
    x = LSTM(units=units,
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(X_encoded)
    # add attention#
    r = Attention(32)(x)  # units need to be changed
    print('r:', r)
    # get the output
    out = Dense(output_dim, activation="sigmoid")(r)
    print('out.shape:', out.shape)  # the dimension should be the number of variables
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model







































def DARNN_TPA(input_dim,time_steps,num_units_LSTM):    # use DARNN's input attention and TPA's temporal attention
    output_dim = input_dim
    filters = 32
    units = num_units_LSTM * 2
    filter_size = 3
    inp=tf.keras.Input(shape=(time_steps,input_dim))
    # convolution layer
    X = Conv1D(filters=filters, kernel_size=filter_size, strides=1, padding="causal")(inp)
    # print('X.shape:',X.shape)       # (?, 32, 512)
    # DA-LSTM Layer
    encoder=Encoder(time_steps,num_units_LSTM)
    X_encoded=encoder(X)   # generates the new input X_tilde for encoder
    # print('X_encoded.shape[1]:',X_encoded.shape[1])
    x=LSTM(units=units,
           kernel_initializer="random_uniform",
           bias_initializer="zeros",
           kernel_regularizer=regularizers.l2(0.001),
           return_sequences=True)(X_encoded)
    # get the 1~t-1 and t hidden state
    H = Lambda(lambda x: x[:, :-1, :])(x)
    ht = Lambda(lambda x: x[:, -1, :])(x)
    ht = Reshape((units, 1))(ht)
    # get the HC by 1*1 convolution
    HC = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)    #################
    print('HC.shape:', HC.shape)  # HC.shape: (?, 128, 31)
    HC = Conv1D(filters=filters - 1, kernel_size=units, strides=1, padding="causal")(HC)
    print('HC2.shape:', HC.shape)  # (?, 128, 31)
    score_mat = CalculateScoreMatrix(units)(HC)
    score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])  # get the score
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat)
    attn_mat = Multiply()([HC, score_mat])  # attn_mat=HC*score_mat
    attn_vec = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat)
    # get the final prediction
    wvt = Dense(units=filters * 4, activation=None)(attn_vec)
    wht = Dense(units=filters * 4, activation=None)(Flatten()(ht))
    yht = Multiply()([wht, wvt])
    # get the output
    out = Dense(output_dim, activation="sigmoid")(yht)
    print('out.shape:',out.shape)                    # the dimension should be the number of variables
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model




def SEnet_TPA(input_dim, time_steps, num_units_LSTM):  # (3,16,16)       # the result is not good
    output_dim = input_dim
    units = num_units_LSTM * 2
    filters=32
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:', inp.shape)  # inp.shape: (?, 16, 3)

    x = LSTM(units=units,  # units=num_units_LSTM
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(inp)
    # get the 1~t-1 and t hidden state
    H = Lambda(lambda x: x[:, :, :])(x)
    print('H1.shape:', H.shape)  # H1.shape: (?, 16, 32)
    ht = Lambda(lambda x: x[:, -1, :])(x)
    ht = Reshape((units, 1))(ht)
    print('ht.shape', ht.shape)  # ht.shape (?, 32, 1)
    ####################################
    # reshape 2D H(t,m) into 2D H(m,t),m represents dimension
    H = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
    print('H2.shape:', H.shape)  # H2.shape: (?, 32, 16)
    # reshape 2D H into 3D H, so that can operate SENet, channels C is dimension m
    print('math.sqrt(time_steps):', math.sqrt(time_steps))  # math.sqrt(time_steps): 4.0
    s = int(math.sqrt(time_steps))
    if math.floor(math.sqrt(time_steps) + 0.5) == math.sqrt(time_steps):  # is Z or not
        H = Reshape((s, s, units))(H)
        print('ok')
    else:
        H = Reshape(math.ceil(math.sqrt(time_steps)), math.ceil(math.sqrt(time_steps)), units)(
            H)  # channels C=units=dimensions,H=W=math.sqrt(time_steps)
        print('no')
        # H=tf.pad(H,)
    print('H3.shape:', H.shape)  # H3.shape: (?, 4, 4, 32)
    # use SENet
    H_se = Squeeze_excitation_layer(input_x=H, out_dim=units, ratio=2, layer_name='squeeze_exc_layer')
    print('H_se.shape:', H_se.shape)  # H_se.shape: (?,4,4,32)
    H_after_se = Reshape((units, time_steps))(H_se)  # from 3D to 2D
    print('H_after_se.shape:', H_after_se.shape)  # H_after_se.shape: (?, 32,16)
    # Vt = Lambda(lambda x: K.sum(x, axis=-1))(H_after_se)  # from H_se to H_after_se,just from array3D to 2D
    # print('Vt.shape:', Vt.shape)  # Vt.shape: (?, 32)
    #####################
    # add TPA part
    print('H.shape:', H.shape)  # H.shape: (?, 32, 29)
    HC = Conv1D(filters=filters, kernel_size=1, strides=1, padding="causal")(H_after_se)
    print('HC2.shape:', HC.shape)  # (?, 32, 32)
    score_mat = CalculateScoreMatrix(units)(HC)  # goal is to make the dimension be equal
    score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])  # get the score
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat)

    attn_mat = Multiply()([HC, score_mat])
    attn_vec = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat)
    # get the final prediction
    wvt = Dense(units=filters, activation=None)(attn_vec)  ##########filters*4
    print('wvt.shape:', wvt.shape)
    wht = Dense(units=filters, activation=None)(Flatten()(ht))
    print('wht.shape:', wht.shape)
    yht = Multiply()([wht, wvt])
    print('yht.shape:', yht.shape)
    # yht = Add()([wht, wvt])   ######################
    # get the output
    out = Dense(output_dim, activation="sigmoid")(yht)
    print('out.shape:', out.shape)  # out.shape: (?, 250)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model



def ConvAtt_TPA(input_dim,time_steps,num_units_LSTM):   #  operate input attention on HC ,too late and have no difference
    output_dim = input_dim
    # filters = num_units_LSTM
    filters = 32
    units = num_units_LSTM * 2
    filter_size = 3
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:',inp.shape)       # (?, 32, 250)
    # convolution layer
    X = Conv1D(filters=filters, kernel_size=filter_size, strides=1, padding="causal")(inp)
    print('X.shape:',X.shape)   # X.shape: (?, 32, 64)
    # DA-LSTM Layer
    encoder = Encoder(time_steps, num_units_LSTM)
    X_encoded = encoder(X)  # generates the new input X_tilde for encoder
    # print('X_encoded.shape[1]:',X_encoded.shape[1])
    # LSTM layer
    x = LSTM(units=units,
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(X_encoded)
    # get the 1~t-1 and t hidden state
    H = Lambda(lambda x: x[:, :-1, :])(x)
    ht = Lambda(lambda x: x[:, -1, :])(x)
    ht = Reshape((units, 1))(ht)
    # get the HC by 1*1 convolution
    HC = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
    print('HC.shape:',HC.shape)        #  HC.shape: (?, 128, 31)
    HC = Conv1D(filters=time_steps - 1, kernel_size=units, strides=1, padding="causal")(HC)   ####
    print('HC2.shape:', HC.shape)  # (?, 128, 31)
    # HC should be updated in myself way( with darnn's input attention)###################
    # HC=Lambda(lambda x:tf.transpose(x,[0,2,1]))(HC)
    # print('ok')
    encoder2=Encoder(num_units_LSTM*2,time_steps-1)
    HC=encoder2(HC)
    print('HC3.shape',HC.shape)      # HC3.shape (?, 128, 31)
    print('ok')
    # HC=Lambda(lambda x:tf.transpose(x,[0,2,1]))(HC)
    # print('ok')
    score_mat = CalculateScoreMatrix(units)(HC)
    score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])  # get the score
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat)
    attn_mat = Multiply()([HC, score_mat])  # attn_mat=HC*score_mat
    attn_vec = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat)
    # get the final prediction
    wvt = Dense(units=filters * 4, activation=None)(attn_vec)
    wht = Dense(units=filters * 4, activation=None)(Flatten()(ht))
    yht = Multiply()([wht, wvt])          ##############
    # get the output
    out = Dense(output_dim, activation="sigmoid")(yht)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model




def DualConv1_TPA(input_dim,time_steps,num_units_LSTM):
    # first operate conv on hidden states to get spatio and then operate conv to get temporal
    # have no difference with tpa
    output_dim=input_dim
    filters=32
    units=num_units_LSTM*2
    filter_size=3

    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:', inp.shape)  # inp.shape: (?, 32, 250)
    # convolution layer
    x=Conv1D(filters=filters,kernel_size=filter_size,strides=1,padding="causal")(inp)
    print('x.shape:',x.shape)    # x.shape: (?, 32, 32)

    # LSTM layer
    x = LSTM(units=units,
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(x)
    # get the 1~t-1 and t hidden state
    H = Lambda(lambda x: x[:, :-1, :])(x)  # H.shape: (?, 31, 128)
    print('H.shape:', H.shape)
    ht = Lambda(lambda x: x[:, -1, :])(x)
    ht = Reshape((units, 1))(ht)
    print('ht.shape', ht.shape)  # ht.shape (?, 128, 1)
    # get the HC by 1*1 convolution
    HC=H   # HC.shape=H.shape=(?, 31, 128)
    HC=Conv1D(filters=filters,kernel_size=units,strides=1,padding="causal")(HC)   # spatio
    print('HC.shape:',HC.shape)
    HC = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
    print('HC.shape:', HC.shape)  # HC.shape: (?, 128, 31)
    HC = Conv1D(filters=filters, kernel_size=time_steps-1, strides=1, padding="causal")(HC)
    print('HC2.shape:', HC.shape)  # (?, 128, 31)
    score_mat = CalculateScoreMatrix(units)(HC)
    print('score_mat.shape:', score_mat.shape)  # score_mat.shape: (?, 128, 128)
    score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])  # get the score
    print('score_mat2.shape:', score_mat.shape)  # score_mat2.shape: (?, 128, 1)
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat)
    print('score_mat3.shape:', score_mat.shape)  # score_mat3.shape: (?, 128, 1)
    attn_mat = Multiply()([HC, score_mat])  # attn_mat=HC*score_mat
    print('attn_mat.shape:', attn_mat.shape)  # attn_mat.shape: (?, 128, 31)
    attn_vec = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat)
    print('attn_vec.shape:', attn_vec.shape)  # attn_vec.shape: (?, 128)
    # get the final prediction
    wvt = Dense(units=filters * 4, activation=None)(attn_vec)
    print('wvt.shape:', wvt.shape)
    wht = Dense(units=filters * 4, activation=None)(Flatten()(ht))
    print('wht.shape:', wht.shape)  # wht.shape: (?, 128)
    yht = Multiply()([wht, wvt])
    print('yht.shape:', yht.shape)  # yht.shape: (?, 128)
    # yht = Add()([wht, wvt])   ######################
    # get the output
    out = Dense(output_dim, activation="sigmoid")(yht)
    print('out.shape:', out.shape)  # out.shape: (?, 250)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model




def DualConv2_TPA(input_dim,time_steps,num_units_LSTM):
    # operate two Conv on hidden state(both spatio and temporal weigths)
    output_dim=input_dim
    filters=32
    units=num_units_LSTM*2
    filter_size=3

    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:', inp.shape)  # inp.shape: (?, 32, 250)
    # convolution layer
    x=Conv1D(filters=filters,kernel_size=filter_size,strides=1,padding="causal")(inp)
    print('x.shape:',x.shape)    # x.shape: (?, 32, 32)

    # LSTM layer
    x = LSTM(units=units,
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(x)
    # get the 1~t-1 and t hidden state
    H = Lambda(lambda x: x[:, :-1, :])(x)
    print('H.shape:', H.shape)      # H.shape: (?, 31, 128)
    ht = Lambda(lambda x: x[:, -1, :])(x)
    ht = Reshape((units, 1))(ht)
    print('ht.shape', ht.shape)  # ht.shape (?, 128, 1)
    # get the HC by 1*1 convolution
    HC_1=Conv1D(filters=filters,kernel_size=units,strides=1,padding="causal")(H)   # spatio
    print('HC.shape:',HC_1.shape)
    score_mat_1 = CalculateScoreMatrix(units)(HC_1)
    print('score_mat.shape:', score_mat_1.shape)  # score_mat.shape: (?, 128, 128)
    score_mat_1 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat_1, ht])  # get the score
    print('score_mat2.shape:', score_mat_1.shape)  # score_mat2.shape: (?, 128, 1)
    # get the attn matrix
    score_mat_1 = Activation("sigmoid")(score_mat_1)
    print('score_mat3.shape:', score_mat_1.shape)  # score_mat3.shape: (?, 128, 1)
    attn_mat_1 = Multiply()([HC_1, score_mat_1])  # attn_mat=HC*score_mat
    print('attn_mat.shape:', attn_mat_1.shape)  # attn_mat.shape: (?, 128, 31)
    attn_vec_1 = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat_1)
    print('attn_vec.shape:', attn_vec_1.shape)  # attn_vec.shape: (?, 128)
    # get the final prediction
    wvt_1 = Dense(units=filters * 4, activation=None)(attn_vec_1)
    print('wvt.shape:', wvt_1.shape)

    HC_2 = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)  # just transpose the H into H, does not operate convolution
    print('HC.shape:', HC_2.shape)  # HC.shape: (?, 128, 31)
    HC_2 = Conv1D(filters=filters - 1, kernel_size=units, strides=1, padding="causal")(HC_2)
    print('HC2.shape:', HC_2.shape)  # (?, 128, 31)
    score_mat_2 = CalculateScoreMatrix(units)(HC_2)
    print('score_mat.shape:', score_mat_2.shape)  # score_mat.shape: (?, 128, 128)
    score_mat_2 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat_2, ht])  # get the score
    print('score_mat2.shape:', score_mat_2.shape)  # score_mat2.shape: (?, 128, 1)
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat_2)
    print('score_mat3.shape:', score_mat_2.shape)  # score_mat3.shape: (?, 128, 1)
    attn_mat_2 = Multiply()([HC_2, score_mat_2])  # attn_mat=HC*score_mat
    print('attn_mat_2.shape:', attn_mat_2.shape)  # attn_mat.shape: (?, 128, 31)
    attn_vec_2 = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat_2)
    print('attn_vec_2.shape:', attn_vec_2.shape)  # attn_vec.shape: (?, 128)
    # get the final prediction
    wvt_2 = Dense(units=filters * 4, activation=None)(attn_vec_2)
    print('wvt_2.shape:', wvt_2.shape)  # wvt.shape: (?, 128)

    wvt=Add()([wvt_1,wvt_2])

    wht = Dense(units=filters * 4, activation=None)(Flatten()(ht))
    print('wht.shape:', wht.shape)  # wht.shape: (?, 128)
    yht = Multiply()([wht, wvt])
    print('yht.shape:', yht.shape)  # yht.shape: (?, 128)
    # yht = Add()([wht, wvt])   ######################
    # get the output
    out = Dense(output_dim, activation="sigmoid")(yht)
    print('out.shape:', out.shape)  # out.shape: (?, 250)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model





def TCTA_TPA(input_dim ,time_steps, num_units_LSTM):
    # # two conv on H and two attention operation on HC
    # self.epochs = epochs
    # self.feat_dim = None
    # feat_dim=input_dim
    # self.input_dim = None
    # input_dim=time_steps
    # self.output_dim=None
    output_dim = input_dim   #
    # self.lr = learning_rate
    filters = 32                       # num_units_LSTM
    # self.batch_size = batch_size
    units=num_units_LSTM*2            #######################
    # self.model_name = "TPA-LSTM"
    filter_size=3
    # batch_size=32
    inp=tf.keras.Input(shape=(time_steps,input_dim))
    print('inp.shape:',inp.shape)  # inp.shape: (?, 30, 250)
    # convolution layer
    # x=Conv1D(filters=filters,kernel_size=filter_size,strides=1,padding="causal")(inp)
    # print('x.shape:',x.shape)    # x.shape: (?, 32, 32)
    # LSTM layer
    x=LSTM(units=units,
           # kernel_initializer="glorot_uniform",
           kernel_initializer="random_uniform",
           bias_initializer="zeros",
           kernel_regularizer=regularizers.l2(0.001),
           return_sequences=True)(inp)
    # get the 1~t-1 and t hidden state
    print('x.shape:',x.shape)              # (?,30,128)
    H=Lambda(lambda x:x[:,:-1,:])(x)
    print('H.shape:',H.shape)              # H.shape: (?, 29, 128)
    ht=Lambda(lambda x:x[:,-1,:])(x)
    ht=Reshape((units,1))(ht)
    print('ht.shape',ht.shape)             # ht.shape (?, 128, 1)
    ####################
    # to operate conv on H columns
    H=Conv1D(filters=units,kernel_size=1,strides=1,padding="causal")(H)  # operate on H(t,m)
    print('1H.shape:',H.shape)      # 1H.shape: (?, 29, 128)
    # get the HC by 1*1 convolution
    H = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)       # from H(t,m) reshape to H(m,t) tor conv
    print('2H.shape:',H.shape)              # H.shape: (?, 128, 29)
    HC=Conv1D(filters=filters,kernel_size=1,strides=1,padding="causal")(H)   # operate on H(m,t)
    print('HC.shape:',HC.shape)            # HC.shape: (?, 128, 32)
    score_mat = CalculateScoreMatrix(units)(HC)         # goal is to make the dimension be equal
    print('1score_mat.shape:',score_mat.shape)     # 1score_mat.shape: (?, 128, 128)
    score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])  # get the score
    print('2score_mat.shape:',score_mat.shape)     # 2score_mat.shape: (?, 128, 1)
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat)
    print('3score_mat.shape:',score_mat.shape)     # 3score_mat.shape: (?, 128, 1)
    attn_mat = Multiply()([HC, score_mat])
    print('attn_mat.shape:',attn_mat.shape)        # attn_mat.shape: (?, 128, 32)
    ################################################
    # chongfu step
    score_mat = CalculateScoreMatrix(units)(attn_mat)  # goal is to make the dimension be equal
    print('1score_mat.shape:', score_mat.shape)  # 1score_mat.shape: (?, 128, 128)
    score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])  # get the score
    print('2score_mat.shape:', score_mat.shape)  # 2score_mat.shape: (?, 128, 1)
    # get the attn matrix
    score_mat = Activation("sigmoid")(score_mat)
    print('3score_mat.shape:', score_mat.shape)  # 3score_mat.shape: (?, 128, 1)
    attn_mat2 = Multiply()([HC, score_mat])
    print('attn_mat.shape:', attn_mat2.shape)  # attn_mat.shape: (?, 128, 32)
    #################################################

    attn_vec = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat2)
    print('attn_vec.shape:',attn_vec.shape)        # attn_vec.shape: (?, 128)
    # get the final prediction
    wvt = Dense(units=filters * 4, activation=None)(attn_vec)
    print('wvt.shape:',wvt.shape)          # wvt.shape: (?, 128)
    wht = Dense(units=filters * 4, activation=None)(Flatten()(ht))
    print('wht.shape:',wht.shape)   # wht.shape: (?, 128)
    yht=Multiply()([wht, wvt])
    print('yht.shape:',yht.shape)      # yht.shape: (?, 128)
    # yht = Add()([wht, wvt])   ######################
    # get the output
    out = Dense(output_dim, activation="sigmoid")(yht)
    print('out.shape:',out.shape)         # out.shape: (?, 250)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model





def Conv_SEnet(input_dim,time_steps,num_units_LSTM):   # (3,16,16)
    output_dim = input_dim
    units = num_units_LSTM*2
    filters=32
    inp = tf.keras.Input(shape=(time_steps, input_dim))
    print('inp.shape:', inp.shape)    # inp.shape: (?, 16, 3)
    print(inp)
    # spatio correlation, transform 1D into 2D,and padding with zero
    ##############################################
    # m=math.ceil(math.sqrt(input_dim))
    # inp0=Reshape([time_steps,m,m])(inp)    # reshape first,,  the question is how to padding with zero
    # print(inp0)
    # print('1inp0.shape:',inp0.shape)   # (?, 16, 2, 2)
    # # inp=tf.pad(inp,[[0,0],[1,1],[1,1]],constant_values=0)
    # # print('2inp.shape:',inp.shape)
    # inp00=Reshape([m,m,time_steps])(inp0)

    #####################################################
    m=math.ceil(math.sqrt(input_dim))
    print('m:',m)
    n=int(math.pow(m,2)-input_dim)   # n represents the number of zeros to be padded
    print('n:',n)
    inp0=tf.pad(inp,[[0,0],[0,0],[0,n]],constant_values=0)    # padding n zeros in dim=2
    print('inp.shape:',inp0.shape)    # inp has been padded
    # to reshape inp from 2D to 3D
    inp00=Reshape([time_steps,m,m])(inp0)
    print(inp00)
    print('inp00.shape:', inp00.shape)        # inp00.shape: (?, 16, 2, 2)
    # inp000=tf.pad(inp00,[[0,0],[1,1],[1,1]],constant_values=0)
    # print('inp000.shape:',inp000.shape)
    inp000 = Reshape([m, m, time_steps])(inp00)

    #####################################################
    inp5=Conv2D(filters=time_steps,kernel_size=[2,m],strides=[1,1],padding="same",data_format='channels_last')(inp000)
    print('ok')
    print('inp5.shape',inp5.shape)
    # inp=Reshape([-1,-1,inp[3],inp[2]])(inp)
    # print('3inp.shape',inp.shape)
    print('inp5.shape[1]',inp5.shape[1])
    print('inp5.shape[2]',inp5.shape[2])
    inp6=Reshape([time_steps,inp5.shape[1]*inp5.shape[2]])(inp5)
    print('inp6.shape:',inp6.shape)
    # input data into LSTM
    # temporal correlation
    x = LSTM(units=units,        # units=num_units_LSTM
             # kernel_initializer="glorot_uniform",
             kernel_initializer="random_uniform",
             bias_initializer="zeros",
             kernel_regularizer=regularizers.l2(0.001),
             return_sequences=True)(inp6)
    # get the 1~t-1 and t hidden state
    H = Lambda(lambda x: x[:, :, :])(x)
    print('H1.shape:', H.shape)        # H1.shape: (?, 16, 32)
    ht = Lambda(lambda x: x[:, -1, :])(x)
    ht = Reshape((units, 1))(ht)
    print('ht.shape', ht.shape)  # ht.shape (?, 32, 1)
    ####################################
    # reshape 2D H(t,m) into 2D H(m,t),m represents dimension
    H = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
    print('H2.shape:',H.shape)              # H2.shape: (?, 32, 16)
    # reshape 2D H into 3D H, so that can operate SENet, channels C is dimension m
    print('math.sqrt(time_steps):',math.sqrt(time_steps))    # math.sqrt(time_steps): 4.0
    #########################################
    # time_steps must can be sqrt
    s=int(math.sqrt(time_steps))
    if math.floor(math.sqrt(time_steps)+0.5)==math.sqrt(time_steps):     # is Z or not
        H=Reshape((s,s,units))(H)
        print('ok')
    else:
        H=Reshape(math.ceil(math.sqrt(time_steps)),math.ceil(math.sqrt(time_steps)),units)(H)   # channels C=units=dimensions,H=W=math.sqrt(time_steps)
        print('no')
        # H=tf.pad(H,)
    print('H3.shape:',H.shape)        # H3.shape: (?, 4, 4, 32)
    ############################################################
    # # solve the problem:the number of time_steps can be any
    # s=math.ceil(math.sqrt(time_steps))
    # t=int(math.pow(s,2)-time_steps)
    # H1=tf.pad(H,[[0,0],[0,0],[0,t]],constant_values=0)
    # print('H1.shape:',H1.shape)
    # H2=Reshape((s,s,units))(H1)
    # print('H.shape:',H2.shape)

    #############################################################
    # use SENet
    H_se=Squeeze_excitation_layer(input_x=H,out_dim=units,ratio=4,layer_name='squeeze_exc_layer')
    print('H_se.shape:',H_se.shape)         # H_se.shape: (?,4,4,32)
    H_after_se=Reshape((units,int(math.pow(s,2))))(H_se)    # from 3D to 2D          Reshape(units,time_steps)
    print('H_after_se.shape:',H_after_se.shape)        # H_after_se.shape: (?, 32,16)
    Vt= Lambda(lambda x: K.sum(x, axis=-1))(H_after_se)       # from H_se to H_after_se,just from array3D to 2D
    print('Vt.shape:',Vt.shape)         # Vt.shape: (?, 32)

    # wht = Dense(units=units, activation=None, name='1dense')(Flatten()(ht))
    # print('wht.shape:', wht.shape)  # wht.shape: (?, 32)
    # yht = Multiply()([wht, Vt])
    # print('yht.shape:', yht.shape)  # yht.shape: (?, 32)
    out = Dense(output_dim, activation="sigmoid", name='2dense')(Vt)
    print('out.shape:', out.shape)  # out.shape: (?, 3)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model





def mLSTM(input_dim, time_steps, num_units_LSTM):
    #  to divide two situations:1.without attention 2.with attention
    # (mul) model1 success
    model=Sequential()
    model.add(LSTM(num_units_LSTM,activation='relu',return_sequences=False,input_shape=(time_steps,input_dim)))
    # model.add(Dropout(0.2))
    # model.add(LSTM(num_units_LSTM,activation='relu',return_sequences=False))      # output_shape=(,32,64)
    # model.add(Dropout(0.2))
    # model.add(AttentionLayer())     #############
    model.add(Dense(input_dim))     # output one value with features=8      (8,)
    model.summary()
    return model



    # # (mul) model1 success
    # model=Sequential()
    # model.add(LSTM(num_units_LSTM,batch_input_shape=(None,input_dim,num_units_dense),return_sequences=True,activation='relu'))
    # model.add(LSTM(num_units_LSTM,return_sequences=False,activation='relu'))
    # # model.add(Conv1D(filters=32, kernel_size=1, strides=1))
    # # model.add(AttentionLayer(attention_size=None))
    # model.add(Dense(num_units_dense))
    # model.add(BatchNormalization())
    # model.summary()
    # return model


    # # (mul) extended model1   32*8     [8 32 8]
    # inputs = keras.Input(shape=(time_steps,input_dim))
    # # lstm_inputs=Permute([2,1])(inputs)
    # lstm_out1 = LSTM(num_units_LSTM, return_sequences=True,activation='relu')(inputs)
    # # lstm_out1=Dropout(0.2)(lstm_out1)
    # # lstm_out1=BatchNormalization()(lstm_out1)
    # lstm_out2 = LSTM(num_units_LSTM, return_sequences=False,activation='relu')(lstm_out1)
    # # lstm_out2 = Dropout(0.2)(lstm_out2)
    # # lstm_out2=BatchNormalization()(lstm_out2)
    # # x = Conv1D(filters=32, kernel_size=1, strides=1)(lstm_out4)  # (?,32,64)   Conv should be operate on rows, instead of columns
    # # x = Dropout(0.3)(x)
    # attention=Dense(input_dim,activation='sigmoid',name='attention_vec')(lstm_out2)
    # attention=Activation('softmax',name='attention_weight')(attention)
    # lstm_out2=Multiply()([lstm_out2,attention])
    # # output = AttentionLayer(attention_size=None)(lstm_out4)
    # outputs = Dense(input_dim, activation='linear')(lstm_out2)
    # model = keras.Model(inputs=inputs, outputs=outputs)
    # model.summary()
    # return model








def mbiGRU(input_dim, num_units_biGRU, num_units_dense):

    # # (mul) model1
    # inputs=keras.Input(shape=(input_dim,num_units_dense))
    # print('inputs.shape:\n', inputs.shape)    # (?,32,8)
    # # x=Conv2D(filters=32,kernel_size=1,strides=(2,2))(inputs)
    # # x=Dropout(0.3)(x)
    # gru_out1=Bidirectional(GRU(num_units_biGRU,return_sequences=True))(inputs)
    # print('gru_out1.shape:\n', gru_out1.shape)     # (?,?,32)
    # gru_out2=Bidirectional(GRU(num_units_biGRU,return_sequences=True))(gru_out1)
    # print('gru_out2.shape:\n', gru_out2.shape)     # (?,?,32)
    # gru_out2=Dropout(0.3)(gru_out2)
    # print('gru_out2.shape after dropout:\n', gru_out2.shape)    # (?,?,32)
    # output=AttentionLayer(attention_size=None)(gru_out2)
    # # inputs (?,32)
    # # H (?,32,32)
    # # score (?,32,1)
    # # output (?,32)
    # # outputs (?,32)
    # print('output.shape:\n', output.shape)         # (?,32)
    # outputs=Dense(num_units_dense,activation='relu')(output)
    # print('outputs.shape:\n', outputs.shape)       # (?,8)
    # model=keras.Model(inputs=inputs,outputs=outputs)
    # model.summary()
    # return model




    # # (mul) extended model1   32*8
    # inputs = keras.Input(shape=(input_dim,num_units_dense))
    # gru_out1 = Bidirectional(GRU(num_units_biGRU, return_sequences=True))(inputs)
    # gru_out2 = Bidirectional(GRU(num_units_biGRU, return_sequences=True))(gru_out1)
    # gru_out2 = Dropout(0.3)(gru_out2)
    # x = Conv1D(filters=32, kernel_size=1, strides=1)(gru_out2)      # (?,32,64)   Conv should be operate on rows, instead of columns
    # x = Dropout(0.3)(x)
    # output = AttentionLayer(attention_size=None)(x)
    # outputs = Dense(num_units_dense, activation='relu')(output)
    # model = keras.Model(inputs=inputs, outputs=outputs)
    # model.summary()
    # return model




    # (mul) extended model1   32*8
    inputs = keras.Input(shape=(input_dim, num_units_dense))
    gru_out1 = GRU(num_units_biGRU, return_sequences=True)(inputs)
    gru_out2 = GRU(num_units_biGRU, return_sequences=True)(gru_out1)
    gru_out2 = Dropout(0.3)(gru_out2)
    print(gru_out2.shape)
    # gru_out2=np.transpose(gru_out2)
    x = Conv1D(filters=32, kernel_size=1, strides=1)(gru_out2)  # (?,32,64)   Conv should be operate on rows, instead of columns
    x = Dropout(0.3)(x)
    output = AttentionLayer(attention_size=None)(x)
    outputs = Dense(num_units_dense, activation='relu')(output)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model





def sLSTM(input_dim,time_steps, num_units_LSTM):    # [1,32,64]

    model = Sequential()
    model.add(LSTM(num_units_LSTM, activation='relu', return_sequences=True, input_shape=(time_steps, input_dim)))
    model.add(Dropout(0.2))
    model.add(LSTM(num_units_LSTM, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))       # output one value with feature=1
    model.summary()
    return model


    #
    # model=Sequential()
    # # model.add(Reshape((input_dim,1),input_shape=(input_dim,)))
    # model.add(LSTM(num_units_LSTM, batch_input_shape=(None, input_size, num_units_dense), return_sequences=True,activation='relu'))
    # model.add(LSTM(num_units_LSTM,return_sequences=False,activation='relu'))
    # # model.add(Conv1D(filters=32, kernel_size=1, strides=1))
    # # model.add(AttentionLayer(attention_size=None))
    # model.add(Dense(num_units_dense,activation='relu'))
    # model.summary()
    # return model





def sbiGRU(input_dim, num_biGRU_layers, num_units_biGRU, num_units_dense):  # num_units_dense=n_features?

    # # original (single) biGRU model codes
    # assert num_biGRU_layers > 0
    # model = Sequential()
    # model.add(Reshape((input_dim,1), input_shape=(input_dim,)))  # input_shape=(input_dim,)  1D data
    # for i in range(num_biGRU_layers - 1):
    #         model.add(Bidirectional(GRU(num_units_biGRU,return_sequences=True)))
    # model.add(Bidirectional(GRU(num_units_biGRU,return_sequences=False)))
    # model.add(Dense(num_units_dense, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(1))
    # return model



    # # (single)simple model1
    # assert num_biGRU_layers > 0
    # # model=Sequential()
    # inputs = keras.Input(shape=(input_dim, 1))
    # # x = Conv1D(filters=64, kernel_size=1)(inputs)  # , padding = 'same'
    # # x = Dropout(0.3)(x)
    # # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # gru_out1 = Bidirectional(GRU(num_units_biGRU, return_sequences=True))(inputs)
    # gru_out2 = Bidirectional(GRU(num_units_biGRU, dropout=0.3, return_sequences=True))(gru_out1)
    # # gru_out2 = Dropout(0.3)(gru_out2)
    # x = AttentionLayer(attention_size=None)(gru_out2)
    # outputs = Dense(1, activation='relu')(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)
    # model.summary()
    # return model



    # # (single)extended model1
    # assert num_biGRU_layers>0
    # inputs=keras.Input(shape=(input_dim,1))
    # gru_out1=Bidirectional(GRU(num_units_biGRU,return_sequences=True))(inputs)
    # gru_out2=Bidirectional(GRU(num_units_biGRU,return_sequences=True))(gru_out1)
    # gru_out2=Dropout(0.3)(gru_out2)
    # x = Conv1D(filters=64, kernel_size=1)(gru_out2)
    # x = Dropout(0.3)(x)
    # output=AttentionLayer(attention_size=None)(x)
    # outputs=Dense(1,activation='relu')(output)
    # model=keras.Model(inputs=inputs,outputs=outputs)
    # model.summary()
    # return model



    # (single)extended model1  1
    assert num_biGRU_layers>0
    inputs=keras.Input(shape=(input_dim,1))
    gru_out1=Bidirectional(GRU(num_units_biGRU,return_sequences=True))(inputs)
    gru_out2=Bidirectional(GRU(num_units_biGRU,return_sequences=True))(gru_out1)
    gru_out2=Dropout(0.3)(gru_out2)
    x = Conv1D(filters=32, kernel_size=1)(gru_out2)    # exist problems when kernel size=2
    x = Dropout(0.3)(x)
    output=AttentionLayer(attention_size=None)(x)
    outputs=Dense(1,activation='relu')(output)
    model=keras.Model(inputs=inputs,outputs=outputs)
    model.summary()
    return model






"""
Inplementation of Attention Layer
"""


class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]  # 32
        hidden_size = input_shape[2]  # 64
        if self.attention_size is None:
            self.attention_size = hidden_size  # hidden_size = input_shape[2] = num_uits_lstm

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.Wp = self.add_weight(name='att_wp', shape=(hidden_size, self.attention_size),
                                  initializer='uniform', trainable=True)
        self.Wx = self.add_weight(name='att_wx', shape=(hidden_size, self.attention_size),
                                  initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        print('inputs.shape:', inputs)  # inputs.shape:Tensor("lstm_2/transpose_1:0",shape=(?,32,64),dtype=float32)
        print('inputs:', inputs)
        step = 32
        self.V = K.reshape(self.V, (-1, 1))  # reshape to one column
        print('self.V.shape:', (self.V).shape)  # (64,1)
        # print('inputs.shape:',inputs.shape)        # (?,?,64)
        print('inputs', inputs[:, step - 1, :].shape)  # (?,64)
        h = inputs[:, step - 1, :]  # the encoder's last hidden state, the decoder's first hidden state
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        print('H', H.shape)  # (?,32,64)
        score = K.softmax(K.dot(H, self.V), axis=1)
        print('score', score.shape)  # (?,32,1)
        output = K.sum(score * inputs, axis=1)
        print('output', output.shape)  # (?,32)
        outputs = K.tanh(K.dot(output, self.Wp) + K.dot(h, self.Wx))
        print('outputs', outputs.shape)  # (?,32)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]





