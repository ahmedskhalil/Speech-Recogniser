#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Ahmed Khalil"
__date__        = "13.10.2019"
__version__     = "0.0.1"
__status__      = "Experimental"

"""
    Record and save audibles 
    as wav files.

"""

import wave
from tqdm import tqdm
from pyaudio import PyAudio, paInt16

FRAMERATE   = 16000
NSAMPLES    = 2000
CHANNELS    = 1
SAMPLEWIDTH = 2
TIME        = 10
CHUNK       = 2014

filename    = '001.wav' 

def save_file(filename, data):

    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(SAMPLEWIDTH)
    wavfile.setframerate(FRAMERATE)
    wavfile.writeframes(b"".join(data))
    wavfile.close()

def record():

    pyaud = PyAudio()
    stream = pyaud.open(format              = paInt16,
                        channels            = CHANNELS,
                        rate                = FRAMERATE,
                        input               = True,
                        frames_per_buffer   = NSAMPLES)

    buff = []; count = 0
    with tqdm(total=8*TIME, desc="recording", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        while count<(8*TIME):
            audiostring = stream.read(NSAMPLES)
            buff.append(audiostring)
            pbar.update(1)
            count += 1

    save_file(filename, buff)
    stream.close()
    print("Finished recording.")


def play():

    wavfile = wave.open(filename, 'rb')
    pyaud   = PyAudio()
    stream  = pyaud.open(format      = pyaud.get_format_from_width(wavfile.getsampwidth()),
                        channels    = wavfile.getnchannels(),
                        rate        = wavfile.getframerate(),
                        output      = True)
    while True:
        data = wavfile.readframes(CHUNK)
        if data=="": break
        stream.write(data)
    stream.close()
    pyaud.terminate()
    print("Finished playing.")


if(__name__=='__main__'):
    record()
    # play()
