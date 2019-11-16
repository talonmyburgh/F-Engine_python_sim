# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:13:40 2018
@author: talonmyburgh
"""
import numpy as np
from fixpoint import fixpoint, cfixpoint
from pfb_coeff_gen import coeff_gen

# =============================================================================
# Bit reversal algorithms used for the iterative fft's
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
def make_fix_twiddle(N,bits,fraction,method="ROUND"):
    twids = cfixpoint(bits,fraction, method = method)
    twids.from_complex(np.exp(-2*np.arange(N//2)*np.pi*1j/N))
    return twids

"""Natural order in DIT FFT that accepts the data, the twiddle factors
(must be bit reversed), a shift register, the bitwidth and fraction
bit width to process at, the twiddle factor bits and allows for staging"""
def iterffft_natural_DIT(DATA,twid,swreg,bits,fraction,twidfrac,staged=False):

    data=DATA.copy()
    N = data.data.shape[0]                                                     #how long is data stream
    
    if(type(swreg)==int):                                           #if integer is parsed rather than list
        shiftreg = [int(x) for x in bin(swreg)[2:]]
        if (len(shiftreg)<int(np.log2(N))):
            for i in range(int(np.log2(N))-len(shiftreg)):
                shiftreg.insert(0,0)
    elif(type(swreg)==list and type(swreg[0])==int):             #if list of integers is parsed
        shiftreg = swreg
    else:
        raise ValueError('Shiftregister must be type int or binary list of ints')
    
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
            jlast = jfirst + distance - 1                                      #first index plus used to index whole group
            W=twid[k]

            slc1 = slice(jfirst,jlast+1,1)
            slc2 = slice(jfirst+distance, jlast+1+distance,1)
            tmp = W * data[slc2]
            tmp >> twidfrac                                                    #slice off lower bit growth from multiply (caused by fraction only)
            tmp.bits =bits                                                     
            tmp.fraction=fraction                                              #fraction will = (frac1+frac2) - hence right shift by frac2
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
# like CASPER does. 
# =============================================================================   

class FixPFB(object):
        """This function takes point size, how many taps, whether to integrate 
        the output or not, what windowing function to use, whether you're 
        running dual polarisations, what rounding and overflow scheme to use,
        fwidth and whether to stage."""
        def __init__(self, N, taps, bits_in, frac_in, bits_fft, frac_fft, 
                     bits_out, frac_out, twidbits, twidfrac, swreg, 
                     bitsofacc=32, fracofacc=31, unsigned = False, 
                     chan_acc =False, datasrc = None, w = 'hann',
                     firmethod="ROUND", fftmethod="ROUND", dual = False,
                     fwidth=1, staged = False):
            
            """Populate PFB object properties"""
            self.N = N                                                         #how many points
            self.chan_acc = chan_acc                                           #if summing outputs
            self.dual = dual                                                   #whether you're processing dual polarisations
            self.taps = taps                                                   #how many taps
            self.bitsofacc = bitsofacc                                         #how many bits to grow to in integration
            self.fracofacc = fracofacc
            self.bits_in = bits_in                                             #input data bitlength
            self.frac_in = frac_in
            self.bits_fft = bits_fft                                           #fft data bitlength
            self.frac_fft = frac_fft
            self.bits_out = bits_out                                           #what bitlength out you want
            self.frac_out = frac_out
            self.fwidth = fwidth                                               #normalising factor for fir window
            if(type(swreg)==int):                                           #if integer is parsed rather than list
                self.shiftreg = [int(x) for x in bin(swreg)[2:]]
                if (len(self.shiftreg)<int(np.log2(N))):
                    for i in range(int(np.log2(N))-len(self.shiftreg)):
                        self.shiftreg.insert(0,0)
            elif(type(swreg)==list and type(swreg[0])==int):             #if list of integers is parsed
                self.shiftreg = swreg
            else:
                raise ValueError('Shiftregister must be type int or binary list of ints')
                
            self.unsigned = unsigned                                           #only used if data parsed in is in a file                                          
            self.staged = staged                                               #whether to record fft stages
            self.twidbits = twidbits                                           #how many bits to give twiddle factors
            self.twidfrac = twidfrac
            self.firmethod=firmethod                                           #rounding scheme in firs
            self.fftmethod=fftmethod                                           #rounding scheme in fft
            
            #Define variables to be used:
            self.reg_real = fixpoint(self.bits_in, self.frac_in,unsigned = self.unsigned,
                                     method = self.firmethod)
            self.reg_real.from_float(np.zeros([N,taps],dtype = np.int64))      #our fir register size filled with zeros orignally
            self.reg_imag = self.reg_real.copy()

            if(datasrc is not None and type(datasrc)==str):                    #if input data file is specified
                self.inputdata = cfixpoint(self.bits_in, self.frac_in,unsigned = self.unsigned,
                           method = self.firmethod)
                self.inputdatadir = datasrc
                self.outputdatadir = datasrc[:-4]+"out.npy"
                self.inputdata.from_complex(np.load(datasrc, mmap_mode = 'r'))
            else:
                self.inputdatadir = None
            
            #the window coefficients for the fir filter
            self.window = fixpoint(self.bits_fft, self.frac_fft,unsigned = self.unsigned,
                                   method = self.firmethod)
            tmpcoeff,self.firsc = coeff_gen(self.N,self.taps,w,self.fwidth)
            self.window.from_float(tmpcoeff)  
            
            #the twiddle factors for the natural input fft
            self.twids = make_fix_twiddle(self.N,self.twidbits,twidfrac,
                                          method=self.fftmethod)
            self.twids = bitrevfixarray(self.twids,self.twids.data.size)
            
        """Takes data segment (N long) and appends each value to each fir.
        Returns data segment (N long) that is the sum of fircontents*window"""
        def _FIR(self,x):
            #push and pop from FIR register array
            self.reg_real.data = np.column_stack(
                    (x.real.data,self.reg_real.data))[:,:-1]
            self.reg_imag.data = np.column_stack(
                    (x.imag.data,self.reg_imag.data))[:,:-1]
            
            X_real = self.reg_real*self.window                                 #compute real and imag products
            X_imag = self.reg_imag*self.window
            prodgrth = X_real.fraction - self.frac_fft                         #-1 since the window coeffs have -1 less fraction
            X = cfixpoint(real = X_real.sum(axis=1),imag = X_imag.sum(axis =1))
            X >> prodgrth +self.firsc                                          #remove growth
            X.bits = self.bits_fft                                             #normalise to correct bit and frac length
            X.fraction = self.frac_fft
            X.normalise()
            X.method = self.fftmethod                                          #adjust so that it now uses FFT rounding scheme
            
            return X                                                           #FIR output
        
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
            self.G_k >> 1                                                      #for bit growth from addition
            self.G_k.bits = self.bits_fft
            self.G_k.normalise()
    
            self.H_k =cfixpoint(real = I_k + I_kflip, imag = R_kflip - R_k)
            self.H_k >> 1
            self.H_k.bits = self.bits_fft
            self.H_k.normalise()
            

        """Here we take the power spectrum of the outputs. Chan_acc dictates
        if one must sum over all outputs produced."""        
        def _pow(self,X):
            if (self.chan_acc):                                                #if accumulation specified
                tmp = X.power()                                                # X times X*
                pwr = X.copy()
                pwr.bits = self.bitsofacc
                pwr.frac=self.fracofacc
                pwr.normalise()                                                #normalise multiplication
                pwr.data = np.sum(tmp.data,axis=1)                             #accumulate                                
                return pwr
            else:                                                              #if no accumulation specified
                pwr = X.power()
                pwr.bits = self.bitsofacc
                pwr.frac=self.fracofacc
                pwr.normalise()                                                #normalise multiplication
                return pwr

        """Here one parses a data vector to the PFB to run. Note it must be
        cfixpoint type if a data file was not specified before"""
        def run(self,DATA, cont = False):
            
            if (DATA is not None):                                             #if a data vector has been parsed
                if(self.bits_in != DATA.bits):
                    raise ValueError("Input data must match precision specified"
                                     +"for input data with bits_in")
                self.inputdata = DATA
            elif(self.inputdata is None):                                      #if no data was specified at all
                raise ValueError ("No input data for PFB specified.")

            size = self.inputdata.data.shape[0]                                #get length of data stream which should be multiple of N
            data_iter = size//self.N                                              #how many cycles of commutator
            
            X = cfixpoint(self.bits_fft, self.frac_fft,unsigned = self.unsigned,
                          method = self.fftmethod)
            
            if(self.staged):                                                   #if all stages need be stored
                X.from_complex(np.empty((self.N,data_iter,int(np.log2(self.N))+2),
                                        dtype = np.complex64))                 #will be tapsize x datalen/point x fft stages +2 
                                                                               #(input and re-ordererd output)  
                for i in range(0,data_iter):                                   #for each data_iter, populate all firs, and run FFT once
                    if(i == 0):
                        X[:,i,:] = iterffft_natural_DIT(self._FIR(self.inputdata[0:self.N]),
                        self.twids,self.shiftreg.copy(),self.bits_fft,self.frac_fft,
                        self.twidfrac,self.staged)
                    else:
                        X[:,i,:] = iterffft_natural_DIT(
                                self._FIR(self.inputdata[i*self.N:i*self.N+self.N]),
                                self.twids,self.shiftreg.copy(),self.bits_fft,
                                self.frac_fft, self.twidfrac, self.staged)
                        
            else:                                                              #if stages don't need to be stored
                X.from_complex(np.empty((self.N,data_iter),
                                        dtype = np.complex64))                 #will be tapsize x datalen/point
                for i in range(0,data_iter):                                   #for each stage, populate all firs, and run FFT once
                    if(i == 0):
                        X[:,i] = iterffft_natural_DIT(self._FIR(self.inputdata[0:self.N]),
                        self.twids,self.shiftreg.copy(),self.bits_fft,self.frac_fft,
                        self.twidfrac, self.staged)

                    else:
                        X[:,i] = iterffft_natural_DIT(
                                self._FIR(self.inputdata[i*self.N:i*self.N+self.N]),
                                self.twids,self.shiftreg.copy(),self.bits_fft,
                                self.frac_fft, self.twidfrac, self.staged)
                
            """Requantise if bitsout<bitsfft"""    
            if(self.bits_out<self.bits_fft):                                   
                X>>(self.bits_fft-self.bits_out)
                X.bits=self.bits_out
                X.fraction = self.frac_out
                X.normalise()
#            
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