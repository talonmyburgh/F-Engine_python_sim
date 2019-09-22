# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:21:43 2018

@author: talonmyburgh
"""
import numpy as np
import numba as nb
from pfb_coeff_gen import coeff_gen

# =============================================================================
# Bit reversal algorithms used for the iterative fft's
# =============================================================================
"""Arranges chronological values in an array in a bit reversed fashion"""
@nb.jit(nopython=True)
def bit_rev(a, bits):
    a_copy = a.copy()
    sz = a.shape[0]
    N = 1<<bits
    for i in range(1,bits):
        a >>=1
        a_copy <<=1
        for j in range(sz):    
            a_copy[j] |= (a[j]&1)
    for k in range(sz):       
        a_copy[k] &= N-1
    return a_copy

"""Takes an array of length N which must be a power of two"""
@nb.jit(nopython=False)
def bitrevarray(array,N): 
    bits = int(np.log2(N))                                                     #number of bits to repr numbers in array
    A = np.empty(N,dtype=np.complex64)
    a=np.arange(N,dtype = np.int64)
    revind = bit_rev(a,bits)
    for i in nb.prange(N):
        A[revind[i]] = array[i]
    return A

# =============================================================================
# FFT: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================
"""Generate array of needed twiddles"""
@nb.jit(nopython=True)
def make_twiddle(N):                                                           #generate array of needed twiddles
    i=np.arange(N//2)
    arr = np.exp(-2*i*np.pi*1j/N)
    return arr

"""Natural order in DIT FFT that accepts the data, the twiddle factors
(bit reversed) and allows for staging"""
@nb.jit(nopython=False)
def iterfft_natural_in_DIT(DATA,twid,staged=False):
    data = np.asarray(DATA,dtype = np.complex64)
    N = data.shape[0]                                                          #how long is data stream
    stgd_data = np.zeros((N,int(np.log2(N))+2),dtype = np.complex64)
    
    if(staged):
        for i in range(N):
            stgd_data[i,0] = data[i]
    num_of_groups = 1                                                          #number of groups - how many subarrays are there?
    distance = N//2                                                            #how far between each fft arm?
    stg=1                                                                      #stage counter
    
    while num_of_groups < N:                                                   #basically iterates through stages
        for k in range(num_of_groups):                                         #iterate through each subarray
            jfirst = 2*k*distance                                              #index to beginning of a group
            jlast = jfirst + distance                                      #first index plus offset - used to index whole group
            W=twid[k]
            for j in range(jfirst,jlast):
                jdist = j+distance
                tmp = W*data[jdist]
                data[jdist] = data[j]-tmp
                data[j] = data[j]+tmp
        num_of_groups *=2
        distance //=2
        if(staged):                                                            #if we are recording stages
            for l in range(N):
                stgd_data[l,stg]=data[l]                                       #log each stage data to array
        stg+=1
        
    if(staged):
        for m in range(N):
            stgd_data[m,-1] = bitrevarray(stgd_data[m,-2],N)                   #post bit-reordering for last stage - added as extra stage
        return stgd_data
    else:
        A=bitrevarray(data,N)                                                  #post bit-reordering
        return A

# =============================================================================
# Floating point PFB implementation making use of the natural order in fft
# like SARAO does. 
# =============================================================================
class FloatPFB(object):
        """This function takes point size, how many taps, what percentage of total data to average over,
        to get data from a file or not,what windowing function, whether you're running dual polarisations,
        whether you'd like data from a stage, and if so which stage - stage 0 being the data in"""
        def __init__(self, N, taps, avg = 1, datasrc = None, w = 'hann',dual = False,
                     staged = False, fwidth=1, chan_acc =False):
            self.N = N                                                         #how many points
            self.avg = avg                                                     #what averaging
            self.dual = dual                                                   #whether you're performing dual polarisations or not
            self.reg =np.zeros([N,taps])                                       #our fir register size filled with zeros orignally
            self.inputdatadir = None
            self.staged=staged
            self.fwidth = fwidth
            self.chan_acc = chan_acc
                
            if(datasrc is not None and type(datasrc)==str):                    #if input data file is specified
                self.inputdatadir = datasrc
                self.outputdatadir = datasrc[:-4]+"out.npy"
                self.inputdata  = np.load(datasrc, mmap_mode = 'r')
            else:
                self.inputdata = None
                
            self.window,self.firsc=coeff_gen(N,taps,w,self.fwidth)
                
            self.twids = make_twiddle(self.N)
            self.twids = bitrevarray(self.twids, len(self.twids))              #for natural order in FFT
                
        """Takes data segment (N long) and appends each value to each fir.
        Returns data segment (N long) that is the sum of fircontents*window"""
        def _FIR(self,x):
            X = np.sum(self.reg*self.window,axis=1) / (2**self.firsc)          #filter
            self.reg = np.column_stack((x,self.reg))[:,:-1]                    #push and pop from FIR register array
            return X
        
        """In the event that that dual polarisations have been selected, we need to 
        split out the data after and return the individual X_k values"""        
        def _split(self,Y_k):
            #reverse the arrays for the splitting function correctly
            R_k = np.real(Y_k)
            R_kflip = R_k.copy()
            R_kflip[1:] = R_kflip[:0:-1]
            
            I_k = np.imag(Y_k)
            I_kflip = I_k.copy()
            I_kflip[1:] = I_kflip[:0:-1]
            
            self.G_k = (1/2)*(R_k+1j*I_k+R_kflip-1j*I_kflip)                   #declares two variables for 2 pols
            self.H_k = (1/2j)*(R_k+1j*I_k-R_kflip+1j*I_kflip)
        
        """Here we take the power spectrum of the outputs. Chan_acc dictates
        if one must sum over all outputs produced."""        
        def _pow(self,X):
            if (self.chan_acc):                                                #if accumulation specified
                pwr = X * np.conj(X)
                pwr = np.sum(pwr,axis=1)
                return pwr
            else:                                                              #if no accumulation specified
                pwr = X * np.conj(X)
                return pwr
                        
        """Given data, (having specified whether the PFB will run in dual or not)
        you parse the data and the PFB will compute the spectrum"""
        def run(self,data=None):
            
            if (data is not None):                                             #if we are using an input data array
                self.inputdata = data
            elif(self.inputdata is None):
                raise ValueError ("No input data for PFB specified.")

            size = self.inputdata.size                                         #get length of data stream
            stages = size//self.N                                              #how many cycles of commutator
            
            if(self.staged):                                                   #if recording staged data
                X = np.zeros((self.N,stages,int(np.log2(self.N))+2),
                             dtype = np.complex64)  
                                                                               #will be tapsize x datalen/point x stages
                for i in range(0,stages):                                      #for each stage, populate all firs, and run FFT once
                    if(i ==0):
                        X[:,i,:] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[0:self.N]),self.twids,
                            self.staged)
                    else:
                        X[:,i,:] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[i*self.N-1:i*self.N+self.N-1]),self.twids,
                            self.staged)
                
            else:                                                              #if not recording staged data
                X = np.zeros((self.N,stages),dtype = np.complex64)
                XFIR = X.copy()
                                                                               #will be tapsize x stages
                for i in range(0,stages):                                      #for each stage, populate all firs, and run FFT once
                    if(i ==0):
                        X[:,i] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[i*self.N:i*self.N+self.N]),
                            self.twids)
                        XFIR[:,i] = self._FIR(self.inputdata[0:self.N])
                    else:
                        X[:,i] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[i*self.N-1:i*self.N+self.N-1]),
                            self.twids)
                        XFIR[:,i] = self._FIR(self.inputdata[i*self.N-1:i*self.N+self.N-1])
                        
            
            """Decide on how to manipulate and display output data"""
            if(self.dual and not self.staged):                                 #If dual processing but not staged                      
                self._split(X)
                self.G_k_pow = self._pow(self.G_k)
                self.H_k_pow = self._pow(self.H_k)
                
            elif(not self.dual and self.staged):                               #If single pol processing and staged
                self.X_k_stgd = X
                self.X_k_pow = self._pow(X[:,:,-1])
                self.X_k = X[:,:,-1]
                
            elif(self.dual and self.staged):                                   #If dual pol and staged
                self.X_k_stgd = X
                self.split(X[:,:,-1])
                self.G_k_pow = self._pow(self.G_k)
                self.H_k_pow = self._pow(self.H_k)
                
            else:                                                              #If single pol and no staging
                self.X_k = X
                self.X_k_pow = self._pow(X)