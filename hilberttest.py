#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:48:41 2019

@author: talon
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

N = 20480
k=np.arange(N)
x_t=np.sin(((2*np.pi/N)*192)*k) + np.sin(((2*np.pi/N)*4881)*k) +np.sin(((2*np.pi/N)*2019)*k)
#check that the real is the hilbert transform of the imaginary on the output of the FFT.

storingarr = np.zeros((512,40),dtype=np.complex64)

for i in range(0,40):
    ind = np.arange(0,512) +(i*512)
    storingarr[:,i] = np.fft.fft(x_t[ind])
    

#plt.plot(np.abs(storingarr[:,5]),'k',np.abs(storingarr[:,15]),'c',np.abs(storingarr[:,30]),'r')
#plt.title("Fig 2: 3 different time outputs of the 512-point FFT")
#plt.show()
#plt.plot(np.abs(storingarr[:,12]))
#plt.show()
print(np.imag(signal.hilbert(np.real(storingarr[12,:]))))
print(np.imag(storingarr[12,:]))


#plt.plot(np.abs(storingarr[300,:]),'k')
#plt.title("Fig 3: FFT result of channel 300 in the 512 FFT result")
#plt.show()
#plt.plot(x_t[:256])
#plt.show()
#print(signal.hilbert(x_t[:15]))
#plt.show()

    
    
    
