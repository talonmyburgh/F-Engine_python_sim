# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:13:40 2018

@author: talonmyburgh
"""
import numpy as np
from fixpoint import fixpoint, cfixpoint
# =============================================================================
# Bit reversal algorithms used for the iterative fft's
# =============================================================================
def bit_rev(a, bits):
    a_copy = a.copy()
    N = 1<<bits
    for i in range(1,bits):
        a >>=1
        a_copy <<=1
        a_copy |= (a[:]&1)
    a_copy[:] &= N-1
    return a_copy

def bitrevfixarray(array,N):                                                   #takes an array of length N which must be a power of two
    bits = int(np.log2(N))                                                     #how many bits it takes to represent all numbers in array
    A = array.copy()
    a = np.arange(N)
    A[bit_rev(a,bits)] = array[:]
    return A

# =============================================================================
# FFT: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================
def make_fix_twiddle(N,bits,fraction,offset=0.0):
    twids = cfixpoint(bits,fraction,offset = offset, method = "ROUND")
    twids.from_complex(np.exp(-2*np.arange(N//2)*np.pi*1j/N))
    return twids

def iterffft_natural_DIT(DATA,twid,shiftreg,bits,fraction,staged=False):       #parse in data,tiddle factors (must be in bit reversed order for natural order in),
                                                                               #how many bits fixpoint numbers are, fraction bits they are, offset, and rounding scheme.
    data=DATA.copy()
    N = data.data.shape[0]                                                     #how long is data stream
    if(staged):
        stgd_data = DATA.copy()
        stgd_data.from_complex(np.zeros((N,int(np.log2(N))+2),
                                        dtype = np.complex64))
        stgd_data[:,0] = data[:]                                                     
    stages = int(np.log2(N))
    if(len(shiftreg)!=stages and type(shiftreg) is not list):
        raise ValueError("Shift register must be of type list, and its length "
                         +"must be log2(data length)")
        
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
            tmp = (W * data[slc2]) >> bits - 1                                 #slice off lower bit growth from multiply
            tmp.bits =bits                                                     #bits will = 2*bits+1 - hence - (bits+1)
            tmp.fraction=fraction                                              #fraction will = 2*(frac1+frac2) - hence - (bits-1)
            tmp.normalise()
            data[slc2] = data[slc1]-tmp
            data[slc1] = data[slc1]+tmp
            
        if shiftreg.pop():                                                     #implement FFT shift and then normalise to correct at end of stage
            data>>1
        data.normalise()
        
        num_of_groups *=2
        distance //=2
        if(staged):                                                            #if we are recording stages
            stgd_data[:,stg]=data[:]                                           #log each stage data to array
        stg+=1
        
    if(staged): 
        stgd_data[:,-1] = bitrevfixarray(stgd_data[:,-2],N)                    #post bit-reordering for last stage - added as extra stage
        return stgd_data
    else:
        return bitrevfixarray(data,N)                                          #post bit-reordering

# =============================================================================
# Floating point PFB implementation making use of the natural order in fft
# like SARAO does. 
# =============================================================================   

class FixPFB(object):
        """This function takes point size, how many taps, what percentage of total data to average over,
        what windowing function, whether you're running dual polarisations and staged (which stage) or not"""
        def __init__(self, N, taps, bits_in, bits_out, shiftreg, unsigned = False,
                     offset = 0.0, chan_acc = 1, datasrc = None, w = 'hanning',dual = False,staged = False):
            self.N = N                                                         #how many points
            self.chan_acc = chan_acc                                           #what averaging
            self.dual = dual                                                   #whether you're performing dual polarisations or not
            self.taps = taps
            self.bits_in = bits_in
            self.bits_out = bits_out
            self.shiftreg = shiftreg
            self.unsigned = unsigned
            self.offset = offset
            self.staged = staged
            
            self.reg_real = fixpoint(self.bits_in, self.bits_in,unsigned = self.unsigned,
                                     offset = self.offset, method = "ROUND_UP")
            self.reg_real.from_float(np.zeros([N,taps],dtype = np.int64))      #our fir register size filled with zeros orignally
            self.reg_imag = self.reg_real.copy()
            
            self.inputdata = cfixpoint(self.bits_in, self.bits_in,unsigned = self.unsigned,
                                       offset = self.offset, method = "ROUND_UP")
            self.inputdatadir = None
            if(datasrc is not None and type(datasrc)==str):                    #if input data file is specified
                self.inputdatadir = datasrc
                self.outputdatadir = datasrc[:-4]+"out.npy"
                self.inputdata.from_complex(np.load(datasrc, mmap_mode = 'r'))
            
            WinDic = {                                                         #dictionary of various filter types
                'hanning' : np.hanning,
                'hamming' : np.hamming,
                'bartlett': np.bartlett,
                'blackman': np.blackman,
                }
            
            self.window = fixpoint(self.bits_out, self.bits_out-1,unsigned = self.unsigned,
                                   offset = self.offset, method = "ROUND_UP")
            self.window.from_float(WinDic[w](taps))     
            self.X_k = None                                                    #our output
                
            self.twids = make_fix_twiddle(self.N,self.bits_out,self.bits_out-1,self.offset)
            self.twids = bitrevfixarray(self.twids,self.twids.data.size)
            
        """Takes data segment (N long) and appends each value to each fir.
        Returns data segment (N long) that is the sum of fircontents*window"""
        def _FIR(self,x):
            X_real = self.reg_real*self.window
            X_imag = self.reg_imag*self.window
            X = cfixpoint(real = X_real.sum(axis=1),imag = X_imag.sum(axis =1))
            X.method = "ROUND_UP"                                              #adjust so that it now uses rounding like in the FFT (unlike FIR)
            X >> (X.bits - self.bits_out)
            X.bits = self.bits_out
            X.fraction = self.bits_out
            X.normalise()
            
            #push and pop from FIR register array
            self.reg_real.data = np.column_stack((x.real.data,self.reg_real.data))[:,:-1]
            self.reg_imag.data = np.column_stack((x.imag.data,self.reg_imag.data))[:,:-1]
            return X
        
        """In the event that that dual polarisations have been selected, we need to 
        split out the data after and return the individual X_k values"""
        def _split(self,Yk):
            #reverse the arrays for the splitting function correctly            
            R_k = Yk.real.copy()
            I_k = Yk.imag.copy()
            
            R_kflip = R_k.copy()
            R_kflip[1:] = R_kflip[:0:-1]
            
            I_kflip = I_k.copy()
            I_kflip[1:] = I_kflip[:0:-1]

            self.G_k = cfixpoint(real = R_k + R_kflip, imag = I_k - I_kflip)   #declares two variables for 2 pols
            self.G_k >> 1
            self.G_k.bits = self.bits_out
            self.G_k.normalise()
    
            self.H_k =cfixpoint(real = I_k + I_kflip, imag = R_kflip - R_k)
            self.H_k >> 1
            self.H_k.bits = self.bits_out
            self.H_k.normalise()
            

        """Here we take the power spectrum of the outputs. The averaging scheme
        tells over what portion of the output data to take the power spectrum of."""        
        def _pow(self,X):
            if(self.chan_acc ==1):                                             #The scheme for averaging used.
                retX = X.real*X.real + X.imag*X.imag
                return retX
            else:
                iterr = int(1/self.chan_acc)
                rng = len(X.data[0,:])//iterr
                Xt = fixpoint(self.bits_out, self.bits_out,unsigned = self.unsigned,
                              offset = self.offset, method = "ROUND")
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
            
            if (DATA is not None):                                             #if we are using an input data array
                if(self.bits_in != DATA.bits):
                    raise ValueError("Input data must match precision specified for input data")
                self.inputdata = DATA
            elif(self.inputdata is None):
                raise ValueError ("No input data for PFB specified.")

            size = self.inputdata.data.shape[0]                                #get length of data stream
            stages = size//self.N                                              #how many cycles of commutator
            
            X = cfixpoint(self.bits_out, self.bits_out,unsigned = self.unsigned,
                          offset = self.offset, method = "ROUND")
            
            if(self.staged):                                                   #if all stages need be stored
                X.from_complex(np.zeros((self.N,stages,int(np.log2(self.N))+2),
                                        dtype = np.complex64))                 #will be tapsize x datalen/point x fft stages +2    
                for i in range(0,stages):                                      #for each stage, populate all firs, and run FFT once
                    if(i == 0):
                        X[:,i,:] = iterffft_natural_DIT(self._FIR(self.inputdata[i*self.N:i*self.N+self.N]),
                        self.twids,self.shiftreg.copy(),self.bits_out,self.bits_out,staged=self.staged)
                    else:
                        X[:,i,:] = iterffft_natural_DIT(self._FIR(self.inputdata[i*self.N-1:i*self.N+self.N-1]),
                         self.twids,self.shiftreg.copy(),self.bits_out,self.bits_out,staged=self.staged)
                        
            else:
                X.from_complex(np.zeros((self.N,stages),dtype = np.complex64)) #will be tapsize x datalen/point
                for i in range(0,stages):                                      #for each stage, populate all firs, and run FFT once
                    if(i == 0):
                        X[:,i] = iterffft_natural_DIT(self._FIR(self.inputdata[i*self.N:i*self.N+self.N]),
                        self.twids,self.shiftreg.copy(),self.bits_out,self.bits_out)
                    else:
                        X[:,i] = iterffft_natural_DIT(self._FIR(self.inputdata[i*self.N-1:i*self.N+self.N-1]),
                         self.twids,self.shiftreg.copy(),self.bits_out,self.bits_out)
                
                
            
            if(self.dual and not self.staged):                                 #decide on how to manipulate data
                self._split(X)
                self.G_k = self._pow(self.G_k)
                self.H_k = self._pow(self.H_k)
            elif(not self.dual and self.staged):
                self.X_k_stgd = X
                self.X_k = self._pow(X[:,:,-1])
            elif(self.dual and self.staged):
                self.X_k = X
                self.split(X[:,:,-1])
                self.G_k = self._pow(self.G_k)
                self.H_k = self._pow(self.H_k)
            else:
                self.X_k = X
            
            if(self.inputdatadir is not None):             
                if(self.dual): 
                    np.save("pol_1_"+self.outputdatadir,self.G_k.to_float())   #save output data as complex (same pol ordering)
                    np.save("pol_2_"+self.outputdatadir,self.H_k.to_float())
                else:   
                    np.save(self.outputdatadir,self.X_k.to_float())
                
