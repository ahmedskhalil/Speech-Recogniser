#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Ahmed Khalil"
__date__        = "13.10.2019"
__version__     = "0.0.1"
__status__      = "Experimental"

"""
    Includes a class ..

"""

#applying runtime check
import platform

class LangModel():
    def __init__(self,modelpath):
        self.modelpath = modelpath()

    def load_model(self):
        self.endict   = 0
        self.model1 = 0
        self.model2 = 0
        self.en     = 0
        model = (self.endict, self.model1, self.model2)
        return model

    def speech_to_text(self, list):
        pass

    
        
        

