#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Ahmed Khalil"
__date__        = "13.10.2019"
__version__     = "0.0.1"
__status__      = "Experimental"

"""
    Includes a class for multi-GPU which:
    - encapsulates model copies for each gpu
    - slices input and send a slice to each 
        model copy.
    - then applies loss to the combined outputs.

"""

import tensorflow as tf, keras
import keras.backend as K, keras.layers as KL

class KParallelModel(keras.models.Model):


    def __init__(self, kerasmodel, gpucount):

        super(KParallelModel, self).__init__()
        self.innermodel = kerasmodel
        self.gpucount   = gpucount
        mergedoutputs   = self.make_parallel()
        super(KParallelModel, self).__init(inputs=self.innermodel.inputs, outputs=mergedoutputs)

    def __getattribute__(self, attributename):
        ''' redirect models '''
        if 'load' in attributename or 'save' in attributename:
            return getattr(self.innermodel, attributename)
        return super(KParallelModel, self).__getattribute__(attributename)

    def summary(self, *args, **kwargs):
        ''' show summary of wrapper & innermodel '''
        super(KParallelModel, self).summary(*args, **kwargs)
        self.innermodel.summary(*args, **kwargs)

    def make_parallel(self):

        inputslices = { name : tf.split(x, self.gpucount) 
                        for name, x in zip(self.innermodel.inputnames, self.innermodel.inputs) }

        outputnames = self.innermodel.outputnames
        outputsall  = []
        for i in range(len(self.innermodel.outputs)):
            outputsall.append([])

        for i in range(self.gpucount):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i): #process slice-wise
                    zippedinputs = zip(self.innermodel.inputnames, self.innermodel.inputs)
                    inputs = [ KL.Lambda(lambda s: inputslices[name][i],
                                            outputshape = lambda s: (None,) + s[1:])(tensor) 
                                                for name, tensor in zippedinputs ]

                    outputs = self.innermodel(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    for l, o in enumerate(outputs):
                        outputsall[l].append(o)

        with tf.device('/cpu:0'):
            mergedout = []
            for outputs, name in zip(outputsall, outputnames):
                def addDimension(tensor):
                    if K.int_shape(tensor) == ():
                        return KL.Lambda(lambda t: K.reshape(t, [1,1]))(tensor)
                    return tensor
                outputs = list(map(addDimension, outputs))

                mergedout.append(KL.Concatenate(axis=0, name=name)(outputs))
        return mergedout

        