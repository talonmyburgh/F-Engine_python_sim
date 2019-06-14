#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:16:52 2019

@author: talon
"""

from timeit import timeit as t

upset = '''from fixpoint import fixpoint
import numba as nb
import numpy as np
from pfb_fixed import bitrevfixarray
fp = fixpoint(10,10)
fp.from_float(np.arange(1024))'''

code = '''bitrevfixarray(fp,1024)'''

print(t(stmt=code,setup=upset,number=10000))