#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Ahmed Khalil"
__date__        = "13.10.2019"
__version__     = "0.0.1"
__status__      = "Experimental"

"""
    Includes a function to parse 
    lines in a dictionary file.

"""



def get_textlines(path):
    
    obj     = open(path, 'r', encoding='utf-8')
    text    = obj.read()
    lines   = text.split('\n')
    symbol  = []
    for i in lines:
        if(i!='' or len(i)>=2):
            l = i.split('\t')
            symbol.append(l[0])
    obj.close()
    symbol.append('_')

    return symbol


if(__name__=='__main__'):
    print(get_textlines('experimental/dictionary.txt')[:3])
