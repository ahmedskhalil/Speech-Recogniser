#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Ahmed Khalil"
__date__        = "10.10.2019"
__version__     = "0.0.1"
__status__      = "Experimental"

"""
    Includes functions related to sound file
    processing.


"""

import math, wave, os, time
import numpy as np, matplotlib.pyplot as plt

from scipy.fftpack import fft

from python_speech_features import mfcc
from python_speech_features import delta

x  = np.linspace(0,400-1,400,dtype=np.int64)
hw = 0.54 - 0.46 * np.cos(2*np.pi*(x)/(400-1))

def read_wav(filename):
    """ read sound files of type wave """
    wavfile      = wave.open(filename, 'rb')
    nframes      = wavfile.getnframes()
    nchannels    = wavfile.getnchannels()
    framerate    = wavfile.getframerate()
    nsamplewidth = wavfile.getsampwidth()
    datastring   = wavfile.readframes(nframes)
    wavfile.close()
    data         = np.fromstring(datastring, dtype=np.short)
    data.shape   = -1, nchannels
    data         = data.T

    return data, framerate


def get_freq_features(wav, samplerate):
    """ get freq features """
    if(samplerate != 16000):
        raise ValueError('Allowed only at 16kHz')

    timewindow = 25
    inputdata  = []

    wavlen    = len(wav[0])
    rangeall  = int(len(wav[0])/samplerate*1000 - timewindow) // 10

    for i in range(0, rangeall):
        pstart   = i * 160
        pend     = pstart + 400
        dataline = []

        for j in range(pstart, pend):
            dataline.append(wav[0][j])


        dataline   = fft(dataline)/wavlen
        dataline_2 = []

        for freqsig in dataline:
            dataline_2.append(freqsig.real)
            dataline_2.append(freqsig)

        inputdata.append(dataline_2[0:len(dataline_2)//2])

    return inputdata


def get_mfcc_features(wav, samplerate):
    f_mfcc   = mfcc(wav[0], samplerate)
    f_mfccd  = delta(f_mfcc, 2)
    f_mfccdd = delta(f_mfccd, 2)
    wav_feat = np.column_stack((f_mfcc, f_mfccd, f_mfccdd))
    
    return wav_feat


def get_freq_features2(wav, samplerate):
    """ get freq features """
    if(samplerate != 16000):
        raise ValueError('Allowed only at 16kHz')

    timewindow = 25
    windowlen  = samplerate/1000*timewindow

    wavarray = np.array(wav)
    wavlen   = wavarray.shape[1]

    rangeall  = int(len(wav[0])/samplerate*1000 - timewindow) // 10
    datainput = np.zeros((1,400), dtype=np.float)
    dataline  = np.zeros((1,400), dtype=np.float)
    for i in range(0, rangeall):
        pstart       = i * 160
        pend         = pstart + 400
        dataline     = wavarray[0,pstart:pend]
        dataline     = np.abs(fft(dataline)) / wavlen
        datainput[i] = dataline[0:200]

    return datainput


def get_freq_features3(wav, samplerate):
    """ get freq features """
    if(samplerate != 16000):
        raise ValueError('Allowed only at 16kHz')

    timewindow = 25
    windowlen  = samplerate/1000*timewindow

    wavarray = np.array(wav)
    wavlen   = wavarray.shape[1]

    rangeall  = int(len(wav[0])/samplerate*1000 - timewindow) // 10
    datainput = np.zeros((rangeall,200), dtype=np.float)
    dataline  = np.zeros((1,400), dtype=np.float)
    for i in range(0, rangeall):
        pstart       = i * 160
        pend         = pstart + 400

        dataline     = wavarray[0,pstart:pend]
        dataline     = dataline * hw
        dataline     = np.abs(fft(dataline)) / wavlen
        datainput[i] = dataline[0:200]

    datainput = np.log(datainput+1)

    return datainput


def get_freq_features4(wav, samplerate):
    """ get freq features """
    if(samplerate != 16000):
        raise ValueError('Allowed only at 16kHz')

    timewindow = 25
    windowlen  = samplerate/1000*timewindow

    wavarray = np.array(wav)
    wavlen   = wavarray.shape[1]

    rangeall  = int(len(wav[0])/samplerate*1000 - timewindow) // 10+1
    datainput = np.zeros((rangeall,windowlen), dtype=np.float)
    dataline  = np.zeros((1,400), dtype=np.float)
    for i in range(0, rangeall):
        pstart       = i * 160
        pend         = pstart + 400

        dataline     = wavarray[0,pstart:pend]
        dataline     = dataline * hw
        dataline     = np.abs(fft(dataline)) / wavlen
        datainput[i] = dataline[0:windowlen // 2]

    datainput = np.log(datainput+1)

    return datainput


def get_scale(energy):

    meanval = energy.mean()
    varival = energy.val()
    res     = (energy-meanval)/np.sqrt(varival)

    return res


def get_scale2(energy):

    maxval = max(energy)
    res    = energy/maxval

    return res


def get_scale3(energy):

    for i in range(len(energy)):
        energy[i] = float(energy[i])/100.0

    return energy


def plot_sound(wav, samplerate):

    time = np.arange(0,len(wav))*(1.0/samplerate)
    plt.plot(time, wav, color='black')
    plt.show()


def get_wavs(filename):
    """ get list of wave files """
    obj         = open(filename, 'r')
    text        = obj.read()
    lines       = text.split('\n')
    filelist    = {}
    wavmark     = []
    for i in lines:
        if(i!=''):
            l = i.split(' ')
            filelist[l[0]] = l[1]
            wavmark.append(l[0])
    obj.close()

    return filelist, wavmark


def get_wavsymbol(filename):
    """ get pronunciation from file """
    obj         = open(filename, 'r')
    text        = obj.read()
    lines       = text.split('\n')
    symbollist  = {}
    symbolmark  = []
    for i in lines:
        if(i!=''):
            l = i.split(' ')
            symbollist[l[0]] = l[1:]
            symbolmark.append(l[0])
    obj.close()

    return symbolmark, symbolmark


if(__name__=='__main__'):

    wavdata, fs = read_wav('experimental/speech.wav')
    plot_sound(wavdata[0], fs)

    t0   = time.time()
    fimg = get_freq_features(wavdata, fs)
    t1   = time.time()
    print("get_freq_features took:", t1-t0)

    t0   = time.time()
    fimg = get_freq_features3(wavdata, fs)
    t1   = time.time()
    print("get_freq_features3 took:", t1-t0)

    fimg = fimg.T
    plt.subplot(111)
    plt.imshow(fimg)
    plt.show
