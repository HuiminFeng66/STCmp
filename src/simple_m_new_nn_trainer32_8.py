
"""
simple1 can be run,but the problem is that maxerror is not a array
the difference between simple1 and simple2 is that maxerror is a array.

the reason of small val_loss and loss may be the difference is too close to zero
we can try to change another dataset
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import argparse
from tensorflow.keras.callbacks import CSVLogger
import models
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store', default=None,dest='train',help='choose training sequence file',required=True)
parser.add_argument('-val', action='store', default=None,dest='val',help='choose validation sequence file',required=True)
parser.add_argument('-model_file', action='store', required = True,dest='model_file',help='weights will be stored with this name')
parser.add_argument('-model_name', action='store', default=None,dest='model_name',help='name of the model to call',required=True)
parser.add_argument('-model_params', action='store',dest='model_params',nargs='+',required=True,help='model parameters (first parameter = past memory used for prediction)', type=int)
parser.add_argument('-log_file', action='store',dest='log_file', default = "log_file",help='Log file')
parser.add_argument('-lr', action='store', type=float,dest='lr', default = 1e-3,help='learning rate for Adam')
parser.add_argument('-noise', action='store', type=float,dest='noise', default = 0.0,help='amount of noise added to X (Unif[-noise,noise])')
parser.add_argument('-epochs', action='store',dest='num_epochs', type = int, default = 20,help='number of epochs to train (if 0, just store initial model))')



def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    print('nrows:\n', nrows)
    print('n:\n', n)
    print('a.strides:\n', a.strides)
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def generate_data(file_path,time_steps):
    series=np.load(file_path)
    print('series.shape:\n',series.shape)   # (325146,8)
    n=series.shape[1]
    print('n:\n',n)               # 8    have 8 variables
    m=series.shape[0]
    print('m:\n',m)               # 325146    have 325146 time-points
    series=series.reshape(-1,n)
    print('new series shape:\n',series.shape)     # (325146,8)
    # series=series.reshape(-1)       # convert into one row
    # data=strided_app(series,time_steps+1,1)


    dataX,dataY=[],[]
    # print('len(data):',len(data))
    for i in range(len(series)-time_steps-1):
        a=series[i:(i+time_steps),:]
        dataX.append(a)
        dataY.append(series[i+time_steps,:])
    TrainX=np.array(dataX)
    Train_Y=np.array(dataY)
    return TrainX,Train_Y


def mse(y_true,y_pred):        #  mean_squared_error
    return K.mean(K.square(y_pred-y_true),axis=-1)



def rmse(y_true,y_pred):            #  root_mean_squared_error
    return K.sqrt(K.mean(K.square(y_pred-y_true),axis=-1))


def mae(y_true,y_pred):      #   mean_absolute_error
    return K.mean(K.abs(y_pred-y_true),axis=-1)


def mape(y_true,y_pred):               # mean_absolute_percentage_error
    diff=K.abs((y_true-y_pred)/K.clip(K.abs(y_true),K.epsilon(),None))
    return 100.*K.mean(diff,axis=-1)


# def rse(y_true,y_pred):
#     rse=np.sqrt(np.sum(np.square(y_true-y_pred))/np.sum(np.square(np.mean(y_true)-y_pred)))
#     return rse


def rse(y_true,y_pred):
    rse=K.sqrt(K.sum(K.square(y_true-y_pred))/K.sum(K.square(K.mean(y_true)-y_pred)))
    return rse


# def corr(y_true,y_pred):
#     m,mp,sig,sigp=y_true.mean(axis=0),y_pred.mean(axis=0),y_true.std(axis=0),y_pred.std(axis=0)
#     corr=((((y_true-m)*(y_pred-mp)).mean(axis=0)/(sig*sigp))[sig!=0]).mean()
#     return corr

d
# def corr(y_true,y_pred):
#     m,mp,sig,sigp=K.mean(y_true,axis=0),K.mean(y_pred,axis=0),K.std(y_true,axis=0),K.std(y_pred,axis=0)
#     corr=K.mean((K.mean((y_true-m)*(y_pred-mp),axis=0)/(sig*sigp)))
#     return corr


def corr(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)



def fit_model(X_train, Y_train, X_val, Y_val, nb_epoch, model):
    optim = tf.keras.optimizers.Adam(lr=arguments.lr)                     ###############
    # rmse=K.mean(K.abs(y_pred-y_true),axis=-1)
    model.compile(loss='mse', optimizer=optim,metrics=['mae',rmse])           # optim
    if nb_epoch == 0:
        model.save(arguments.model_file)
        return
    checkpoint = ModelCheckpoint(arguments.model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False)
    csv_logger = CSVLogger(arguments.log_file, append=True, separator=';')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=3, verbose=1)
    callbacks_list = [checkpoint, csv_logger, early_stopping]
    # global history;
    model.fit(X_train, Y_train, epochs=nb_epoch, verbose=1, shuffle=True, callbacks=callbacks_list,validation_data=(X_val, Y_val))




arguments = parser.parse_args()
print(arguments)


num_epochs=arguments.num_epochs
sequence_length=arguments.model_params[1]   # arguments.model_params[1]=time_steps=32           #####################
X_train,Y_train = generate_data(arguments.train, sequence_length)
# print('X_train:\n',X_train.shape)     # (325113,32,8)
# print('Y_train:\n',Y_train.shape)      # (325113,8)
X_val,Y_val = generate_data(arguments.val, sequence_length)
# print('X_val:\n',X_val.shape)          # (325114,32,8)
# print('Y_val:\n',Y_val.shape)           # (325114,8)
X_train = X_train + np.random.uniform(-arguments.noise,arguments.noise,np.shape(X_train))
X_val = X_val + np.random.uniform(-arguments.noise,arguments.noise,np.shape(X_val))

# # predict the diff rather than absolute value
# print(X_train[0].shape)     # (32,8)
# print(X_train[1].shape)     # (32,8)
# print(X_train[2].shape)     # (32,8)
# print(X_train[325112].shape)    # (32,8)
# print('X_train[:,:,-1].shape:\n',X_train[:,:,-1].shape)      # (325113,32)

# print('X_train[:,-1,:].shape:\n',X_train[:,-1,:].shape)    # (325113,8)
# print('Y_train.shape:\n',Y_train.shape)                # (325113,8)
# print('Y_train[0,:]:',Y_train[0,:])
# print('Y_train[1,:]:',Y_train[1,:])
# print('X_train[0,-1,:]:',X_train[0,-1,:])
# print('X_train[1,-1,:]:',X_train[1,-1,:])
Y_train = Y_train-np.reshape(X_train[:,-1,:],np.shape(Y_train))
# print('Y_train[0,:]:',Y_train[0,:])
# print('Y_train[1,:]:',Y_train[1,:])
Y_val = Y_val-np.reshape(X_val[:,-1,:],np.shape(Y_val))
# print('Y_train:\n',Y_train)
# print('Y_val:\n',Y_val)

model = getattr(models, arguments.model_name)(*arguments.model_params)
fit_model(X_train, Y_train, X_val, Y_val, num_epochs, model)



# def plot(keys,title='title',ylabel='y'):
#     for key in keys:
#         plt.plot(history.history[key])
#         # plt.plot(history.history[key])
#
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.xlabel('epoch')
#     plt.legend(keys,loc='upper left')
#     plt.show()


# plot history
# plot(['loss','val_loss'],'loss')
# plot(['mean_absolute_error','val_mean_absolute_error'],'mean_absolute_error')
# plot(['mean_absolute_percentage_error','val_mean_absolute_percentage_error'],'mean_absolute_percentage_error')