import os, sys 
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import Reshape, Input
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adagrad, Adam, RMSprop, Adamax, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import keras
import hdf5storage
import h5py
import librosa
import numpy as np
import scipy.io.wavfile
import scipy
from os import listdir
from os.path import isfile, isdir, join
import matplotlib.pyplot as plt
import matplotlib
from keras.utils import plot_model
import pydot

# opt = RMSprop(lr=0.0001, rho=0.9 ,epsilon=1e-06)
opt = Nadam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
# n_classes = 2
feature_dim = 257

def wav2spec(wavfile, sr, forward_backward=None, SEQUENCE=None, norm=True, hop_length=256):
    
    sr, y = scipy.io.wavfile.read(wavfile)

    NUM_FRAME = 2  # number of backward frame and forward frame
    NUM_FFT = 512

    D = librosa.stft(y.astype('float32'),
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)

   
    epsilon = np.finfo(float).eps

    D = D + epsilon
    Sxx = np.log10(abs(D) ** 2)
    if norm:
        Sxx_mean = np.mean(Sxx, axis=1).reshape(257, 1)
        Sxx_var = np.var(Sxx, axis=1).reshape(257, 1)
        Sxx_r = (Sxx - Sxx_mean) / Sxx_var
    else:
        Sxx_r = np.array(Sxx)
    idx = 0
    # set data into 3 dim and muti-frame(frame, sample, num_frame)
    if forward_backward:
        Sxx_r = Sxx_r.T
        return_data = np.empty(
            (100000, np.int32(NUM_FRAME * 2) + 1, np.int32(NUM_FFT / 2) + 1))
        frames, dim = Sxx_r.shape

        for num, data in enumerate(Sxx_r):
            idx_start = idx - NUM_FRAME
            idx_end = idx + NUM_FRAME
            if idx_start < 0:
                null = np.zeros((-idx_start, dim))
                tmp = np.concatenate((null, Sxx_r[0:idx_end + 1]), axis=0)
            elif idx_end > frames - 1:
                null = np.zeros((idx_end - frames + 1, dim))
                tmp = np.concatenate((Sxx_r[idx_start:], null), axis=0)
            else:
                tmp = Sxx_r[idx_start:idx_end + 1]

            return_data[idx] = tmp
            idx += 1
        shape = return_data.shape
        if SEQUENCE:
            return return_data[:idx]
        else:
            return return_data.reshape(shape[0], shape[1] * shape[2])[:idx]

    else:
        Sxx_r = np.array(Sxx_r).T
        shape = Sxx_r.shape
        if SEQUENCE:
            return Sxx_r.reshape(shape[0], 1, shape[1])
        else:
            return Sxx_r


def spec2wav(wavfile, output_filename, spec_test, hop_length=None):
    sr, y = scipy.io.wavfile.read(wavfile)  # noisy wav file: for extracting phase and sr

    D = librosa.stft(y.astype('float32'),
                     n_fft=512,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)

    epsilon = np.finfo(float).eps
    D = D + epsilon
    phase = np.exp(1j * np.angle(D))
    Sxx_r_tmp = np.array(spec_test)  # enhanced log power magnitude: spec_test
    Sxx_r_tmp = np.sqrt(10 ** Sxx_r_tmp)
    Sxx_r = Sxx_r_tmp.T
    reverse = np.multiply(Sxx_r, phase)

    result = librosa.istft(reverse,
                           hop_length=hop_length,
                           win_length=512,
                           window=scipy.signal.hann)

    y_out = librosa.util.fix_length(result, len(y), mode='edge')
  
    scipy.io.wavfile.write(output_filename, sr, y_out.astype('int16'))  # save file: save to output_filename
    return y_out


def ddae_generator_noVAD_1(Trpath_1, Trpath_2, Trpath_3, TrClPth , fs=16000, ws=3, Frame_size=512, Frame_shft=256):
    

    lst_ind = 0
    i=0

    while True:
        
        Onydata_1 = wav2spec(Trpath_1[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=True,
                           SEQUENCE=None, norm=True)
        x=axis_x_train[i]*np.ones((1,(np.size(Onydata_1,0))))
        y=axis_y_train[i]*np.ones((1,(np.size(Onydata_1,0))))
        z=axis_z_train[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,x)
        Onydata_1 = insert_feature(Onydata_1,0,y)
        Onydata_1 = insert_feature(Onydata_1,0,z)
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,x1)
        Onydata_1 = insert_feature(Onydata_1,0,y1)
        Onydata_1 = insert_feature(Onydata_1,0,z1)
        Onydata_1 = insert_feature(Onydata_1,0,x2)
        Onydata_1 = insert_feature(Onydata_1,0,y2)
        Onydata_1 = insert_feature(Onydata_1,0,z2)
        Onydata_1 = insert_feature(Onydata_1,0,x3)
        Onydata_1 = insert_feature(Onydata_1,0,y3)
        Onydata_1 = insert_feature(Onydata_1,0,z3)
        d1=distance2ch1_train[i]*np.ones((1,(np.size(Onydata_1,0))))
        d2=distance2ch2_train[i]*np.ones((1,(np.size(Onydata_1,0))))
        d3=distance2ch3_train[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,d1)
        Onydata_1 = insert_feature(Onydata_1,0,d2)
        Onydata_1 = insert_feature(Onydata_1,0,d3)

      


        Onydata_2 = wav2spec(Trpath_2[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=True,
                           SEQUENCE=None, norm=True)
        x=axis_x_train[i]*np.ones((1,(np.size(Onydata_2,0))))
        y=axis_y_train[i]*np.ones((1,(np.size(Onydata_2,0))))
        z=axis_z_train[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,x)
        Onydata_2 = insert_feature(Onydata_2,0,y)
        Onydata_2 = insert_feature(Onydata_2,0,z)
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,x1)
        Onydata_2 = insert_feature(Onydata_2,0,y1)
        Onydata_2 = insert_feature(Onydata_2,0,z1)
        Onydata_2 = insert_feature(Onydata_2,0,x2)
        Onydata_2 = insert_feature(Onydata_2,0,y2)
        Onydata_2 = insert_feature(Onydata_2,0,z2)
        Onydata_2 = insert_feature(Onydata_2,0,x3)
        Onydata_2 = insert_feature(Onydata_2,0,y3)
        Onydata_2 = insert_feature(Onydata_2,0,z3)
        d1=distance2ch1_train[i]*np.ones((1,(np.size(Onydata_2,0))))
        d2=distance2ch2_train[i]*np.ones((1,(np.size(Onydata_2,0))))
        d3=distance2ch3_train[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,d1)
        Onydata_2 = insert_feature(Onydata_2,0,d2)
        Onydata_2 = insert_feature(Onydata_2,0,d3) 

        Onydata_3 = wav2spec(Trpath_3[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=True,
                           SEQUENCE=None, norm=True)
        x=axis_x_train[i]*np.ones((1,(np.size(Onydata_3,0))))
        y=axis_y_train[i]*np.ones((1,(np.size(Onydata_3,0))))
        z=axis_z_train[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,x)
        Onydata_3 = insert_feature(Onydata_3,0,y)
        Onydata_3 = insert_feature(Onydata_3,0,z)
        
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,x1)
        Onydata_3 = insert_feature(Onydata_3,0,y1)
        Onydata_3 = insert_feature(Onydata_3,0,z1)
        Onydata_3 = insert_feature(Onydata_3,0,x2)
        Onydata_3 = insert_feature(Onydata_3,0,y2)
        Onydata_3 = insert_feature(Onydata_3,0,z2)
        Onydata_3 = insert_feature(Onydata_3,0,x3)
        Onydata_3 = insert_feature(Onydata_3,0,y3)
        Onydata_3 = insert_feature(Onydata_3,0,z3)
        d1=distance2ch1_train[i]*np.ones((1,(np.size(Onydata_3,0))))
        d2=distance2ch2_train[i]*np.ones((1,(np.size(Onydata_3,0))))
        d3=distance2ch3_train[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,d1)
        Onydata_3 = insert_feature(Onydata_3,0,d2)
        Onydata_3 = insert_feature(Onydata_3,0,d3)

        Ocndata = wav2spec(TrClPth[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=False,
                           SEQUENCE=None, norm=False)


        i += 1
        lst_ind += 1
        if  i==300:
            i=0
        

        if lst_ind == len(TrClPth):
            lst_ind = 0
        yield np.concatenate((Onydata_1, Onydata_2, Onydata_3), axis=1), Ocndata
        


def ddae_generator_noVAD_2(Vapath_1, Vapath_2, Vapath_3, VaClPth ,  fs=16000, ws=512, Frame_size=512, Frame_shft=256):
    

    lst_ind = 0
    i=0
    

    while True:
        
        Onydata_1 = wav2spec(Vapath_1[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=True,
                           SEQUENCE=None, norm=True)
        x2=axis_x_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        y2=axis_y_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        z2=axis_z_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,x2)   
        Onydata_1 = insert_feature(Onydata_1,0,y2)
        Onydata_1 = insert_feature(Onydata_1,0,z2)
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,x1)
        Onydata_1 = insert_feature(Onydata_1,0,y1)
        Onydata_1 = insert_feature(Onydata_1,0,z1)
        Onydata_1 = insert_feature(Onydata_1,0,x2)
        Onydata_1 = insert_feature(Onydata_1,0,y2)
        Onydata_1 = insert_feature(Onydata_1,0,z2)
        Onydata_1 = insert_feature(Onydata_1,0,x3)
        Onydata_1 = insert_feature(Onydata_1,0,y3)
        Onydata_1 = insert_feature(Onydata_1,0,z3)
        
        d1=distance2ch1_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        d2=distance2ch2_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        d3=distance2ch3_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,d1)
        Onydata_1 = insert_feature(Onydata_1,0,d2)
        Onydata_1 = insert_feature(Onydata_1,0,d3)


        Onydata_2 = wav2spec(Vapath_2[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=True,
                           SEQUENCE=None, norm=True)
        x2=axis_x_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        y2=axis_y_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        z2=axis_z_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,x2) 
        Onydata_2 = insert_feature(Onydata_2,0,y2)
        Onydata_2 = insert_feature(Onydata_2,0,z2)
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,x1)
        Onydata_2 = insert_feature(Onydata_2,0,y1)
        Onydata_2 = insert_feature(Onydata_2,0,z1)
        Onydata_2 = insert_feature(Onydata_2,0,x2)
        Onydata_2 = insert_feature(Onydata_2,0,y2)
        Onydata_2 = insert_feature(Onydata_2,0,z2)
        Onydata_2 = insert_feature(Onydata_2,0,x3)
        Onydata_2 = insert_feature(Onydata_2,0,y3)
        Onydata_2 = insert_feature(Onydata_2,0,z3)
        d1=distance2ch1_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        d2=distance2ch2_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        d3=distance2ch3_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,d1)
        Onydata_2 = insert_feature(Onydata_2,0,d2)
        Onydata_2 = insert_feature(Onydata_2,0,d3)

        Onydata_3 = wav2spec(Vapath_3[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=True,
                           SEQUENCE=None, norm=True)
        x2=axis_x_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        y2=axis_y_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        z2=axis_z_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,x2)
        Onydata_3 = insert_feature(Onydata_3,0,y2)
        Onydata_3 = insert_feature(Onydata_3,0,z2)
        
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,x1)
        Onydata_3 = insert_feature(Onydata_3,0,y1)
        Onydata_3 = insert_feature(Onydata_3,0,z1)
        Onydata_3 = insert_feature(Onydata_3,0,x2)
        Onydata_3 = insert_feature(Onydata_3,0,y2)
        Onydata_3 = insert_feature(Onydata_3,0,z2)
        Onydata_3 = insert_feature(Onydata_3,0,x3)
        Onydata_3 = insert_feature(Onydata_3,0,y3)
        Onydata_3 = insert_feature(Onydata_3,0,z3)
        d1=distance2ch1_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        d2=distance2ch2_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        d3=distance2ch3_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,d1)
        Onydata_3 = insert_feature(Onydata_3,0,d2)
        Onydata_3 = insert_feature(Onydata_3,0,d3)

        Ocndata = wav2spec(VaClPth[lst_ind], hop_length=Frame_shft, sr=fs, forward_backward=False,
                           SEQUENCE=None, norm=False)


        i += 1
        lst_ind += 1
        if  i==20:
            i=0
        

        if lst_ind == len(VaClPth):
            lst_ind = 0
        yield np.concatenate((Onydata_1, Onydata_2, Onydata_3), axis=1), Ocndata

def test(Vapath_1,Vapath_2,Vapath_3,flag,fs=16000, ws=512, Frame_size=512, Frame_shft=256):
    

    i=flag

    while True:
    
        Onydata_1 = wav2spec(Vapath_1, hop_length=Frame_shft, sr=fs, forward_backward=True,
                          SEQUENCE=None, norm=True)
        x=axis_x_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        y=axis_y_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        z=axis_y_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,x)  
        Onydata_1 = insert_feature(Onydata_1,0,y)   
        Onydata_1 = insert_feature(Onydata_1,0,z)
        
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_1,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_1,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,x1)
        Onydata_1 = insert_feature(Onydata_1,0,y1)
        Onydata_1 = insert_feature(Onydata_1,0,z1)
        Onydata_1 = insert_feature(Onydata_1,0,x2)
        Onydata_1 = insert_feature(Onydata_1,0,y2)
        Onydata_1 = insert_feature(Onydata_1,0,z2)
        Onydata_1 = insert_feature(Onydata_1,0,x3)
        Onydata_1 = insert_feature(Onydata_1,0,y3)
        Onydata_1 = insert_feature(Onydata_1,0,z3)
        d1=distance2ch1_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        d2=distance2ch2_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        d3=distance2ch3_test[i]*np.ones((1,(np.size(Onydata_1,0))))
        Onydata_1 = insert_feature(Onydata_1,0,d1)
        Onydata_1 = insert_feature(Onydata_1,0,d2)
        Onydata_1 = insert_feature(Onydata_1,0,d3)
        

        Onydata_2 = wav2spec(Vapath_2, hop_length=Frame_shft, sr=fs, forward_backward=True,
                          SEQUENCE=None, norm=True)
       
        x=axis_x_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        y=axis_y_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        z=axis_z_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,x)  
        Onydata_2 = insert_feature(Onydata_2,0,y)
        Onydata_2 = insert_feature(Onydata_2,0,z)
        
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_2,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_2,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,x1)
        Onydata_2 = insert_feature(Onydata_2,0,y1)
        Onydata_2 = insert_feature(Onydata_2,0,z1)
        Onydata_2 = insert_feature(Onydata_2,0,x2)
        Onydata_2 = insert_feature(Onydata_2,0,y2)
        Onydata_2 = insert_feature(Onydata_2,0,z2)
        Onydata_2 = insert_feature(Onydata_2,0,x3)
        Onydata_2 = insert_feature(Onydata_2,0,y3)
        Onydata_2 = insert_feature(Onydata_2,0,z3)
        d1=distance2ch1_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        d2=distance2ch2_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        d3=distance2ch3_test[i]*np.ones((1,(np.size(Onydata_2,0))))
        Onydata_2 = insert_feature(Onydata_2,0,d1)
        Onydata_2 = insert_feature(Onydata_2,0,d2)
        Onydata_2 = insert_feature(Onydata_2,0,d3)
       

        Onydata_3 = wav2spec(Vapath_3, hop_length=Frame_shft, sr=fs, forward_backward=True,
                          SEQUENCE=None, norm=True)
      
        x=axis_x_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        y=axis_y_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        z=axis_z_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,x)
        Onydata_3 = insert_feature(Onydata_3,0,y)
        Onydata_3 = insert_feature(Onydata_3,0,z)
        
        x1=mic1_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y1=mic1_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z1=mic1_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        x2=mic2_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y2=mic2_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z2=mic2_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        x3=mic3_axis_x[i]*np.ones((1,(np.size(Onydata_3,0))))
        y3=mic3_axis_y[i]*np.ones((1,(np.size(Onydata_3,0))))
        z3=mic3_axis_z[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,x1)
        Onydata_3 = insert_feature(Onydata_3,0,y1)
        Onydata_3 = insert_feature(Onydata_3,0,z1)
        Onydata_3 = insert_feature(Onydata_3,0,x2)
        Onydata_3 = insert_feature(Onydata_3,0,y2)
        Onydata_3 = insert_feature(Onydata_3,0,z2)
        Onydata_3 = insert_feature(Onydata_3,0,x3)
        Onydata_3 = insert_feature(Onydata_3,0,y3)
        Onydata_3 = insert_feature(Onydata_3,0,z3)
        d1=distance2ch1_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        d2=distance2ch2_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        d3=distance2ch3_test[i]*np.ones((1,(np.size(Onydata_3,0))))
        Onydata_3 = insert_feature(Onydata_3,0,d1)
        Onydata_3 = insert_feature(Onydata_3,0,d2)
        Onydata_3 = insert_feature(Onydata_3,0,d3)
        

        
        #print(x)
        #print(y)
        #print(z)
        #print('一一一一一一')
        
        data = np.concatenate((Onydata_1, Onydata_2, Onydata_3), axis=1)
        return data

def insert_feature(Onydata,f,insert_data):
    
    Onydata=np.insert(Onydata, np.size(Onydata,1), values=insert_data, axis=1)
    
    return Onydata



Trpath_1 = []#training
Trpath_2 = []
Trpath_3 = []
Vapath_1 = []#validation
Vapath_2 = []
Vapath_3 = []
TrClPth = []
VaClPth = []


mat_data = scipy.io.loadmat('/workdir/107wei/EX6/coordinate.mat')
source_position =mat_data['coordinate']

#人的位置 train
axis_x_train = []
axis_y_train = []
axis_z_train = []
#人的位置 test
axis_x_test = []
axis_y_test = []
axis_z_test = []

for i in range(320):
    if i<300:
        x=source_position[i,0]
        y=source_position[i,1]
        z=source_position[i,2]
        axis_x_train.append(x)
        axis_y_train.append(y)
        axis_z_train.append(z)
    if i>=300:
        x=source_position[i,0]
        y=source_position[i,1]
        z=source_position[i,2]
        axis_x_test.append(x)
        axis_y_test.append(y)
        axis_z_test.append(z)   

#麥克風的位置
mic1_axis_x=[]
mic1_axis_y=[]
mic1_axis_z=[]
mic2_axis_x=[]
mic2_axis_y=[]
mic2_axis_z=[]
mic3_axis_x=[]
mic3_axis_y=[]
mic3_axis_z=[]

for i in range(320):
    x=2+1*np.cos(90 * np.pi / 180)
    y=3+1*np.sin(90 * np.pi / 180)
    z=0.4
    mic1_axis_x.append(x)
    mic1_axis_y.append(y)
    mic1_axis_z.append(z)
    x=2+1*np.cos(210 * np.pi / 180)
    y=3+1*np.sin(210 * np.pi / 180)
    z=0.4
    mic2_axis_x.append(x)
    mic2_axis_y.append(y)
    mic2_axis_z.append(z)
    x=2+1*np.cos(330 * np.pi / 180)
    y=3+1*np.sin(330 * np.pi / 180)
    z=0.4
    mic3_axis_x.append(x)
    mic3_axis_y.append(y)
    mic3_axis_z.append(z)        

#人和麥克風距離
distance2ch1_train = []
distance2ch2_train = []
distance2ch3_train = [] 
distance2ch1_test = []
distance2ch2_test = []
distance2ch3_test = [] 
for i in range(320):
    if i<=299:
        d21 = np.sqrt((axis_x_train[i]-mic1_axis_x[i])**2 + (axis_y_train[i]-mic1_axis_y[i])**2 + (axis_z_train[i]-mic1_axis_z[i])**2)
        d22 = np.sqrt((axis_x_train[i]-mic2_axis_x[i])**2 + (axis_y_train[i]-mic2_axis_y[i])**2 + (axis_z_train[i]-mic2_axis_z[i])**2)
        d23 = np.sqrt((axis_x_train[i]-mic3_axis_x[i])**2 + (axis_y_train[i]-mic3_axis_y[i])**2 + (axis_z_train[i]-mic3_axis_z[i])**2)
        distance2ch1_train.append(d21)
        distance2ch2_train.append(d22)
        distance2ch3_train.append(d23)
    if i>=300:
        k=i-300
        d21 = np.sqrt((axis_x_test[k]-mic1_axis_x[i])**2 + (axis_y_test[k]-mic1_axis_y[i])**2 + (axis_z_test[k]-mic1_axis_z[i])**2)
        d22 = np.sqrt((axis_x_test[k]-mic2_axis_x[i])**2 + (axis_y_test[k]-mic2_axis_y[i])**2 + (axis_z_test[k]-mic2_axis_z[i])**2)
        d23 = np.sqrt((axis_x_test[k]-mic3_axis_x[i])**2 + (axis_y_test[k]-mic3_axis_y[i])**2 + (axis_z_test[k]-mic3_axis_z[i])**2)
        distance2ch1_test.append(d21)
        distance2ch2_test.append(d22)
        distance2ch3_test.append(d23)


noise_list=['battle016','buccaneer1','cafeteria_babble','n36','n84','PC Fan Noise','pinknoise','發動機噪聲']
#tr
for i in range(-5, 17, 3):  
    db = str(i) + 'db'                           
    for j in range(3):
        ch = 'ch'+str(j+1)
        for k in range(8):
            for l in range(300):
                num = str(l + 1) + '.wav'
                if j == 0:
                    path = '/workdir/107wei/EX6/train noise/'+db+'/'+ch+'/'+noise_list[k]+'/'+num
                    Trpath_1.append(path)
                elif j == 1:
                    path = '/workdir/107wei/EX6/train noise/'+db+'/'+ch+'/'+noise_list[k]+'/'+num
                    Trpath_2.append(path)
                elif j == 2:
                    path = '/workdir/107wei/EX6/train noise/'+db+'/'+ch+'/'+noise_list[k]+'/'+num
                    Trpath_3.append(path)
for i in range(64):               
    for j in range(300):
        num = str(j + 1) + '.wav'
        path = '/workdir/107wei/EX6/target_v4/'+ num
        TrClPth.append(path)


noise_list_2=['Car_Noise_Idle 60mph','SIREN3_WAIL_FAST','Street_Noise_downtown','Water_Cooler']
#ts
for i in range(-10, 16, 5):  
    db = str(i) + 'db'                           
    for j in range(3):
        ch = 'ch'+str(j+1)
        for k in range(4):
            for l in range(20):
                num = str(l + 301) + '.wav'
                if j == 0:
                    path = '/workdir/107wei/EX6/test noise/'+db+'/'+ch+'/'+noise_list_2[k]+'/'+num
                    Vapath_1.append(path)
                elif j == 1:
                    path = '/workdir/107wei/EX6/test noise/'+db+'/'+ch+'/'+noise_list_2[k]+'/'+num
                    Vapath_2.append(path)
                elif j == 2:
                    path = '/workdir/107wei/EX6/test noise/'+db+'/'+ch+'/'+noise_list_2[k]+'/'+num
                    Vapath_3.append(path)
for i in range(24):               
    for j in range(20):
        num = str(j + 301) + '.wav'
        path = '/workdir/107wei/EX6/target_v4/'+ num
        VaClPth.append(path)                       
    
    


ch = Input(shape=( 1300* 3 ,))

hidden1 = Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(ch)

hidden2 = Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(hidden1)

hidden3 = Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(hidden2)

hidden4 = Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(hidden3)

hidden5 = Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(hidden4)

hidden6 = Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(hidden5)

output = Dense(257, activation='linear')(hidden6)
model_1 = Model(inputs=ch, outputs=output)






g1 = ddae_generator_noVAD_1(Trpath_1, Trpath_2, Trpath_3, TrClPth ,  fs=16000, ws=512, Frame_size=512, Frame_shft=256)

g2 = ddae_generator_noVAD_2(Vapath_1, Vapath_2, Vapath_3, VaClPth ,  fs=16000, ws=512, Frame_size=512, Frame_shft=256)

Epoches = 100
model_1.trainable = True
model_1.compile(loss="mse", optimizer=opt, loss_weights=[1], metrics=['accuracy'])
model_1.summary()

# from keras.utils import plot_model
# plot_model(model_1, to_file='model_7.jpg')           

checkpointer = [
ModelCheckpoint(
filepath="./ex6_15dim_lr0.0005.hdf5",
verbose=0,
save_best_only=True,
monitor="val_loss",
mode="min")]

Num_traindata = len(Trpath_1)
Num_validdata = len(Vapath_1)

history = model_1.fit_generator(generator=g1, steps_per_epoch=Num_traindata, epochs=Epoches,             #if test mark
                                 validation_data=g2, validation_steps=Num_validdata, verbose=1,
                                 max_queue_size=1, workers=1, callbacks=checkpointer)

TrainERR = history.history['loss']
ValidERR = history.history['val_loss']
print('Minimun error:%f, at iteration: %i' % (np.min(np.asarray(ValidERR)), np.argmin(np.asarray(ValidERR)) + 1))
print('drawing the training process...')
matplotlib.pyplot.figure(1)
matplotlib.pyplot.plot(TrainERR, 'b', label='TrainERR')
matplotlib.pyplot.plot(ValidERR, 'r', label='ValidERR')
matplotlib.pyplot.legend()
matplotlib.pyplot.xlabel('epoch')
matplotlib.pyplot.ylabel('error')
matplotlib.pyplot.grid(True)
   # matplotlib.pyplot.show()
matplotlib.pyplot.savefig('ex6_15dim_lr0.0005' + '.png', dpi=150)

model_1.load_weights("./ex6_15dim_lr0.0005.hdf5")




flag = 0
count =1
for i in range(len(VaClPth)):
    if flag ==20:
        count += 1
        flag = 0
    Onydata = test(Vapath_1[i],Vapath_2[i],Vapath_3[i],flag,fs=16000, ws=512, Frame_size=512, Frame_shft=256)
    y_pr_1 =model_1.predict(Onydata)
    
    y_pr_wav = spec2wav(VaClPth[i], './ex6_15dim_lr0.0005/'+str(count)+'/' + str((i+ 301)-((count-1)*20)) + '.wav', y_pr_1, hop_length=256)
    flag +=1

