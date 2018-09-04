# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:44:07 2018

@author: User
"""
import numpy as np
from pfb_fixed import bitrevfixarray, make_fix_twiddle
from pfb_floating import bitrevarray, make_twiddle
import matplotlib.pyplot as plt
from fixpoint import cfixpoint
from collections import deque


N = 256
n = np.arange(N)
bits =18
fraction = 18
method = "truncate"

####SIGNALS######
sig1 = np.cos(2*np.pi*n/N)/2.5
sig2 = np.zeros(N)/2.5
sig2[10:20]=1
sig3 = sig1+1j*sig2

fsig1 = cfixpoint(bits,fraction,method = method)
fsig2 = cfixpoint(bits,fraction,method = method)
fsig3 = cfixpoint(bits,fraction,method = method)
fsig1.from_complex(sig1)
fsig2.from_complex(sig2)
fsig3.from_complex(sig3)

####PARAMS####
twidsfloat = make_twiddle(N)
twidsfloat = bitrevarray(twidsfloat,twidsfloat.size)
twidsfix = make_fix_twiddle(N,bits,fraction-1,method = method)
twidsfix = bitrevfixarray(twidsfix,twidsfix.data.size)


#TEST FFT's#
def iterfft_test(s,w,st):
    a = np.asarray(s,dtype = np.complex).copy()
    N = a.size                                  #how long is data stream
    pairs_in_group = N//2                       #how many butterfly pairs per group - starts at 1/2*full data length obviously
    num_of_groups = 1                           #number of groups - how many subarrays are there?
    distance = N//2                             #how far between each fft arm?
    while num_of_groups < st:                    #basically iterates through stages
        for k in range(num_of_groups):          #iterate through each subarray
            jfirst = 2*k*pairs_in_group         #index to beginning of a group
            jlast = jfirst + pairs_in_group - 1 #first index plus offset - used to index whole group
            W=w[k]
            for j in range(jfirst, jlast + 1):
                tmp = W*a[j+distance]
                a[j+distance] = a[j]-tmp
                a[j] = a[j]+tmp
        pairs_in_group //=2
        num_of_groups *=2
        distance //=2
    #A=bitrevarray(a,N)                          #post bit-reordering
    return a  
    
def iterffft_test(d,twid,shiftreg,bits,fraction,st,offset=0.0,method="round"):  #parse in data,tiddle factors (must be in bit reversed order for natural order in),
                                                                                 #how many bits fixpoint numbers are, fraction bits they are, offset, and rounding scheme.
    data=d.copy()
    N = data.data.size                                                           #how long is data stream
    stages = np.log2(N)
    if(type(shiftreg) == deque and len(shiftreg)==stages):
        shiftreg = shiftreg
    elif(type(shiftreg)==list and len(shiftreg)==stages):
        shiftreg = deque(shiftreg)
    else:
        raise ValueError("shift register must be of type list or deque, and its length must be that of log2(data length)")
    
    pairs_in_group = N//2                                                        #how many butterfly pairs per group - starts at 1/2*full data length obviously
    num_of_groups = 1                                                            #number of groups - how many subarrays are there?
    distance = N//2                                                              #how far between each fft arm?
    while num_of_groups < st:                                                    #basically iterates through stages
        for k in range(num_of_groups):                                           #iterate through each subarray
            jfirst = 2*k*pairs_in_group                                          #index to beginning of a group
            jlast = jfirst + pairs_in_group - 1                                  #first index plus offset - used to index whole group
            W=twid[k]
            for j in range(jfirst,jlast + 1):
                tmp = (W * data[j+distance]) >> bits - 1                         #slice off lower bit growth from multiply
                tmp.bits =bits                                                   #bits will = 2*bits+1 - hence - (bits+1)
                tmp.fraction=fraction                                            #fraction will = 2*(frac1+frac2) - hence - (bits-1)
                tmp.normalise()
                
                data[j+distance] = data[j]-tmp
                data[j] = data[j]+tmp
        if shiftreg.pop():                                                       #implement FFT shift and then normalise to correct at end of stage
            data>>1
        data.normalise()
        
        pairs_in_group //=2
        num_of_groups *=2
        distance //=2
    #A=bitrevfixarray(data,N)                                                    #post bit-reordering
    return data

st=1
while st <= N:
    shiftreg = deque([0,1,1,1,1,1,1,1])
    #floatingsave
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=False)
    valsfl = iterfft_test(sig1,twidsfloat,st)
    ax[0,0].plot(np.real(valsfl),'g')
    
    #fixedsave
    valsfx = iterffft_test(fsig1,twidsfix,shiftreg,bits=bits,fraction = fraction,method = method,st=st).to_complex()
    ax[0,1].plot(np.real(valsfx),'r')
    ax[0,0].set_title('REAL, stage: '+str(st))
    
    ax[1,0].plot(np.imag(valsfl),'g')
    ax[1,1].plot(np.imag(valsfx),'r')
    ax[1,0].set_title('IMAG, stage: '+str(st))
    
    ax[2,0].plot(np.abs(valsfl),'g')
    ax[2,1].plot(np.abs(valsfx),'r')
    ax[2,0].set_title('ABS, stage: '+str(st))
    
    fig.savefig('../snapsf_engine/cos/st'+str(st)+'_cos.png')
    fig.clear()
    st=st*2