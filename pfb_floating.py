# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:21:43 2018

@author: talonmyburgh
"""
import numpy as np
from pfb_coeff_gen import coeff_gen

# =============================================================================
# Bit reversal algorithms used for the iterative fft's data re-ordering
# =============================================================================
"""Arranges chronological values in an array in a bit reversed fashion"""
def bit_rev(a, bits):
    a_copy = a.copy()
    N = 1<<bits
    for i in range(1,bits):
        a >>=1
        a_copy <<=1
        a_copy |= (a[:]&1)
    a_copy[:] &= N-1
    return a_copy

"""Takes an array of length N which must be a power of two"""
def bitrevarray(array,N): 
    bits = int(np.log2(N))                                                     #number of bits to repr numbers in array
    A = np.empty(N,dtype=np.complex64)
    a=np.arange(N)
    A[bit_rev(a,bits)] = array[:]
    return A

# =============================================================================
# FFT: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================
"""Generate array of needed twiddles"""
def make_twiddle(N):                                                           
    i=np.arange(N//2)
    arr = np.exp(-2*i*np.pi*1j/N)
    return arr

"""Natural order in DIT FFT that accepts the data, the twiddle factors
(bit reversed) and allows for staging"""
def iterfft_natural_in_DIT(DATA,twid,staged=False):
    data = np.asarray(DATA,dtype = np.complex64)
    N = data.shape[0]                                                          #how long is data stream
    
    if(staged):
        stgd_data = np.zeros((N,int(np.log2(N))+2),dtype = np.complex64)
        stgd_data[:,0] = data[:]
    num_of_groups = 1                                                          #number of groups - how many subarrays are there?
    distance = N//2                                                            #how far between each fft arm?
    stg=1                                                                      #stage counter
    
    while num_of_groups < N:                                                   #basically iterates through stages
        for k in range(num_of_groups):                                         #iterate through each subarray
            jfirst = 2*k*distance                                              #index to beginning of a group
            jlast = jfirst + distance - 1                                      #first index plus offset - used to index whole group
            W=twid[k]
            slc1 = slice(jfirst,jlast+1,1)
            slc2 = slice(jfirst+distance, jlast+1+distance,1)
            tmp = W*data[slc2]
            data[slc2] = data[slc1]-tmp
            data[slc1] = data[slc1]+tmp
        num_of_groups *=2
        distance //=2
        if(staged):                                                            #if we are recording stages
            stgd_data[:,stg]=data[:]                                           #log each stage data to array
        stg+=1
        
    if(staged): 
        stgd_data[:,-1] = bitrevarray(stgd_data[:,-2],N)                       #post bit-reordering for last stage - added as extra stage
        return stgd_data
    else:
        A=bitrevarray(data,N)                                                  #post bit-reordering
        return A

# =============================================================================
# Floating point PFB implementation making use of the natural order in fft
# like CASPER does. 
# =============================================================================

class FloatPFB(object):
        """This function takes point size, how many taps, what percentage of total data to average over,
        to get data from a file or not,what windowing function, whether you're running dual polarisations,
        whether you'd like data from a stage, and if so which stage - stage 0 being the data in"""
        def __init__(self, N, taps, datasrc = None, w = 'hann',dual = False,
                     staged = False, fwidth=1, chan_acc = False):
            self.N = N                                                         #how many points                                                   #what averaging
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
            
            self.window = coeff_gen(N,taps,w,self.fwidth)[0]                   #Get window coefficients and scaling 
                                                                               #factor to use in FIR registers.
            self.twids = make_twiddle(self.N)
            self.twids = bitrevarray(self.twids, len(self.twids))              #for natural order in FFT
                
        """Takes data segment (N long) and appends each value to each fir.
        Returns data segment (N long) that is the sum of fircontents*windowcoeffs"""
        def _FIR(self,x):
            self.reg = np.column_stack((x,self.reg))[:,:-1]                    #push and pop from FIR register array
            X = np.sum(self.reg*self.window,axis=1)                            #filter and scale
            return X
        
        """For dual polarisation processing, we need to split the data after
        FFT and return the individual complex spectra"""        
        def _split(self,Y_k):
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
                pwr = np.real(np.sum(pwr,axis=1))
                return pwr
            else:                                                              #if no accumulation specified
                pwr = np.real((X * np.conj(X)))
                return pwr
                        
        """Here one parses a data vector to the PFB to run. Note it must be
        numpy array of length N if a data file was not specified before"""
        def run(self,data=None):
            
            if (data is not None):                                             #if we are using an input data array
                self.inputdata = data
            elif(self.inputdata is None):
                raise ValueError ("No input data for PFB specified.")

            size = self.inputdata.size                                         #get length of data stream
            stages = size//self.N                                              #how many cycles of commutator
            
            if(self.staged):                                                   #if storing staged data
                X = np.empty((self.N,stages,int(np.log2(self.N))+2),
                             dtype = np.complex64)
                                                                               #will be tapsize x datalen/point x stages
                for i in range(0,stages):                                      #for each stage, populate all firs, and run FFT once
                    if(i ==0):
                        X[:,i,:] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[0:self.N]),self.twids,
                            self.staged)
                    else:
                        X[:,i,:] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[i*self.N:i*self.N+self.N]),self.twids,
                            self.staged)
                
            else:                                                              #if storing staged data
                X = np.empty((self.N,stages),dtype = np.complex64)
                                                                               #will be tapsize x stages
                for i in range(0,stages):                                      #for each stage, populate all firs, and run FFT once
                    if(i == 0):
                        X[:,i] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[0:self.N]),
                            self.twids)
                    else:
                        X[:,i] = iterfft_natural_in_DIT(self._FIR(
                                self.inputdata[i*self.N:i*self.N+self.N]),
                            self.twids)
            
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