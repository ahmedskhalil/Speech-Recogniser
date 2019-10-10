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

def get_mfcc(wav, samplerate):
    """ get mfcc features """
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





