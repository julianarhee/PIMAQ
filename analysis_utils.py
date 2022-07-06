
####!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/07/06 08:54:07
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

Some useful functions used a lot across steps.
'''
import re
import glob
import numpy as np


# ###############################################################
# General
# ###############################################################
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def isnumber(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    except TypeError:
        return False

    return True
