# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:13:40 2018

@author: User
"""
import numpy as np
from fixpoint import fixpoint, cfixpoint

# =============================================================================
# Bit reversal algorithms used for the iterative fft's
# =============================================================================
def bit_rev(a, bits):
    a_copy = a
    N = 1<<bits    
    for i in range(1,bits):
        a >>=1
        a_copy <<=1
        a_copy |= (a&1)
    a_copy &= N-1
    return a_copy

def bitrevfixarray(array,N):
    bits = int(np.log2(N))
    A=array.copy()
    for k in range(0, N):
        A[bit_rev(k,bits)] = array[k]
    return A

# =============================================================================
# FFT: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================
from collections import deque

def make_fix_twiddle(N,bits,fraction,offset=0.0, method="round"):
    twids = cfixpoint(bits,fraction,offset = offset, method = method)
    tmp = cfixpoint(bits,fraction,offset=offset, method = method)
    twids.from_complex(np.zeros(N//2,dtype=np.complex))
    for i in range(0,N//2):
        tmp.from_complex(np.exp(-2*i*np.pi*1j/N)/2.001)
        twids[i] = tmp
    return twids


def iterffft_natural_DIT(a,twid,shiftreg,bits,fraction,offset=0.0,method="round"):  #parse in data,tiddle factors (must be in bit reversed order for natural order in),
                                                                                 #how many bits fixpoint numbers are, fraction bits they are, offset, and rounding scheme.
    data=a.copy()
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
    while num_of_groups < N:                                                     #basically iterates through stages
        for k in range(num_of_groups):                                           #iterate through each subarray
            jfirst = 2*k*pairs_in_group                                          #index to beginning of a group
            jlast = jfirst + pairs_in_group - 1                                  #first index plus offset - used to index whole group
            W=twid[k]
            for j in range(jfirst,jlast+1):
                tmp = (W * data[j+distance]) >> bits
                tmp.bits -= bits + 1
                tmp.fraction -= bits
                tmp.normalise()
                data[j+distance] = data[j]-tmp
                data[j] = data[j]+tmp
        if shiftreg.pop():                                                   #implement FFT shift and then normalise to correct at end of stage
            data>>1
        pairs_in_group //=2
        num_of_groups *=2
        distance //=2
    A=bitrevfixarray(data,N)                                                        #post bit-reordering
    return A

# =============================================================================
# Floating point PFB implementation making use of the natural order in fft
# like SARAO does. 
# =============================================================================
import matplotlib.pyplot as plt    

class FixPFB(object):
        """This function takes point size, how many taps, what percentage of total data to average over,
        what windowing function, whether you're running dual polarisations, and data type"""
        def __init__(self, N, taps, bits, fraction,shiftreg, unsigned = False, offset = 0.5,method = "round",  avgperc = 1,w = 'hanning',dual = False):
            self.N = N                   #how many points
            self.avg = avgperc           #what averaging
            self.dual = dual             #whether you're performing dual polarisations or not
            
            self.bits = bits
            self.fraction = fraction
            self.shiftreg = shiftreg
            self.unsigned = unsigned
            self.offset = offset
            self.method = method
            
            self.reg = cfixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)
            self.reg.from_complex(np.zeros([N,taps])) #our fir register size filled with zeros orignally
            
            WinDic = {                   #dictionary of various filter types
                'hanning' : np.hanning,
                'hamming' : np.hamming,
                'bartlett': np.bartlett,
                'blackman': np.blackman,
                }
            self.window = cfixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)
            self.window.from_complex(WinDic[w](taps))     
            self.X_k = cfixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)                 #our output
            
            if(dual):                       #if dual, our two outputs
                self.H_k = cfixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)
                self.G_k = cfixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)
                
            self.twids = make_fix_twiddle(self.N,self.bits,self.fraction,self.offset,self.method)
            self.twids = bitrevfixarray(self.twids,self.N)
            
        """Takes data segment (N long) and appends each value to each fir.
        Returns data segment (N long) that is the sum of fircontents*window"""
        def _FIR(self,x):
            X =  (self.reg*self.window).sum(axis=1).quantise(self.bits,self.fraction,unsigned=self.unsigned)
            tmp = np.column_stack((x.data,self.reg.data))[:,:-1]
            self.reg.real.data = np.real(tmp)
            self.reg.imag.data = np.imag(tmp)     #push and pop from FIR register array
            return X
        
        """In the event that that dual polarisations have been selected, we need to 
        split out the data after and return the individual X_k values"""        
        def _split(self,Yk):
            length = np.size(Yk.data,axis=1)
            self.H_k.from_complex(np.zeros([self.N,length]))
            self.G_k.from_complex(np.zeros([self.N,length]))
            
            zeros = np.zeros(Yk.data.shape)
            tmpfx = fixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)
            tmpfx.from_float(zeros)
            
            R_k = cfixpoint(real=Yk.real,imag = tmpfx)
            I_k = cfixpoint(real = tmpfx,imag = Yk.imag)
            
            R_kflip = R_k.copy()
            R_kflip[1:] = R_kflip[1:][::-1]
            
            I_kflip = I_k.copy()
            I_kflip[1:] = I_kflip[1:][::-1]
            
            cst2 = cfixpoint(1, 0,unsigned = self.unsigned,offset = self.offset, method = self.method)
            cst3 = cfixpoint(1, 0,unsigned = self.unsigned,offset = self.offset, method = self.method)
            cst2.from_complex(-1j)
            cst3.from_complex(1)

            self.G_k[:,:] = (R_k[:,:]+cst2*I_k[:,:]+R_kflip[:,:]-cst2*I_kflip[:,:]).r_shift(1)
            self.H_k[:,:] = (cst3*cst2*(R_k[:,:]+cst2*I_k[:,:]-R_kflip[:,:]+cst2*I_kflip[:,:])).r_shift(1)

        

        """Here we take the power spectrum of the outputs. The averaging scheme
        tells over what portion of the output data to take the power spectrum of."""        
        def _pow(self,X):
            if(self.avg ==1):
                retX = X.real*X.real + X.imag*X.imag
                return retX
            else:
                iterr = int(1/self.avg)
                rng = len(X.data[0,:])//iterr
                Xt = fixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)
                Xt.from_float(np.zeros([self.N,iterr]))
                for i in range(0,iterr):
                    if(i ==0):
                        xt = X[:,i*rng:i*rng+rng].sum(axis=1)
                        Xt[:,i] = xt.real*xt.real+xt.imag*xt.imag
                    else:
                        xt = X[:,i*rng-1:i*rng+rng-1].sum(axis=1)
                        Xt[:,i] = xt.real*xt.real+xt.imag*xt.imag
                return Xt

        """Given data, (having specified whether the PFB will run in dual or not)
        you parse the data and the PFB will compute the spectrum (continuous data mode to still add)"""
        def run(self,DATA, cont = False):
            if (cont ==False):
                    size = DATA.data.shape[0]                           #get length of data stream
                    stages = size//self.N                               #how many cycles of commutator
                    
                    X = cfixpoint(self.bits, self.fraction,unsigned = self.unsigned,offset = self.offset, method = self.method)
                    X.from_complex(np.zeros([self.N,stages]))           #will be tapsize x stage
                    
                    for i in range(0,stages):                           #for each stage, populate all firs, and run FFT once
                        if(i ==0):
                            X[:,i] = iterffft_natural_DIT(self._FIR(DATA[i*self.N:i*self.N+self.N]),self.bits,self.fraction,self.shiftreg)
                        else:
                            X[:,i] = iterffft_natural_DIT(self._FIR(DATA[i*self.N-1:i*self.N+self.N-1]),self.bits,self.fraction,self.shiftreg)
                    if(self.dual): 
                        self._split(X)
                        self.H_k = self._pow(self.H_k)
                        self.G_k = self._pow(self.G_k)
                    else:   
                        self.X_k = self._pow(X)
            else:
                pass

        """Plotting method to display the spectrum - has option to display input alongside"""
        def show(self,save=False,flnm = 'plot.png'):
            n = np.arange(self.N)
            if(self.dual):
                fig = plt.figure(1)
                plt.subplot(211)
                plt.plot(n,self.H_k.to_float())
            
                plt.subplot(212)
                plt.plot(n,self.G_k.to_float())
                if(save): fig.savefig(flnm)
                
            else:
                fig = plt.plot(n,self.X_k.to_float())
                if(save): fig.savefig(flnm)
                
