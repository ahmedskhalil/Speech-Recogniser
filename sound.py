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
import numpy as np, matplotlib as plt

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

# print(read_wav('experimental/speech.wav')[0:50])

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
        datainput[i] = dataline[0:200]

    return datainput

# data, fs = read_wav('experimental/speech.wav')
# print(get_freq_features2(data, 16000))

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



