# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:10:59 2019

load dataset bbox distribution, perform low pass filter and nonlinear regression

@author: Wen Wen
"""

import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt

class histRegression():
    def __init__(self, dist_name='Normal',*arg):
        self.dist_name=dist_name
        self.params=arg

def butterworth_LP_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = scisig.lfilter(b, a, data)
    
    # plot the filter frequncy responce
    w, h = scisig.freqz(b, a, worN=200)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.show()
    
    return y

def getMeanStd(data_x,data_y):
    # calculate mean and variance, the x and y should have same length
    datay=[int(i.real) for i in data_y]
    datax=data_x
    elist=[] # element list
    for x,y in zip(datax,datay):
        elist+=[x]*y
    
    elist=np.array(elist)
    mean=elist.mean()
    std=elist.std()
    return mean, std, elist

def FFT(data):
    return np.fft.fft(data)

def IFFT(data):
    return np.fft.ifft(data)

def freqCutOff(data,cutoff=0.3,ftype='lowpass'):
    # data is complex numbers after FFT
    cutlow=int(len(data)*0.5*cutoff)
    cuthigh=int(len(data)*(1-0.5*cutoff)) # floor
    output=[]
    if ftype=='lowpass':
        for i,d in zip(range(len(data)),data):
            if i<cutlow or i>cuthigh:
                output.append(data[i])
            else:
                output.append(complex(0))
        return np.array(output)
    elif ftype=='highpass':
        for i,d in zip(range(len(data)),data):
            if i>cutlow and i<cuthigh:
                output.append(data[i])
            else:
                output.append(complex(0))
        return np.array(output)

def loadHist(loadpath):
    data=np.array(np.load(loadpath))
    return {'y':data[0],
            'x':data[1]}

if __name__=='__main__':
    loadpath='./aspect_ratio.npy'
    order=3 # order of filter
    fs=30 #sample rate, in Hertz
    cutoff = 3  # desired cutoff frequency of the filter, Hz
    
    hist=loadHist(loadpath)
    
    signal=hist['y']
    signal_butterLP=butterworth_LP_filter(signal,cutoff,fs,order=order)

    signal_FFT=FFT(signal)
    signal_cutoff=freqCutOff(signal_FFT,0.2,'lowpass')
    signal_IFFT=IFFT(signal_cutoff)
    
    plt.subplot(2,1,1)
    plt.title('FFT of signal before (up) and after (down) LP filtering')
    plt.plot(signal_FFT)
    plt.subplot(2,1,2)
    plt.plot(signal_cutoff)
    plt.show()
    
    # plot signal after butterworth low pass filter
    plt.title('Low Pass Filtering')
    plt.plot(hist['x'][:-1],signal,'b-')
    plt.plot(hist['x'][:-1],signal_IFFT,'r-')
    plt.show()
    
    mean, std, elist = getMeanStd(hist['x'][:-1],signal_IFFT)
    print('mean={}, std={}'.format(mean,std))
    
    
    
""" End of file """