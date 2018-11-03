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
    a_copy = a.copy()
    N = 1<<bits
    for i in range(1,bits):
        a >>=1
        a_copy <<=1
        a_copy |= (a[:]&1)
    a_copy[:] &= N-1
    return a_copy

def bitrevfixarray(array,N): #takes an array of length N which must be a power of two
    bits = int(np.log2(N)) #how many bits it takes to represent all numbers in array
    A = array.copy()
    a = np.arange(N)
    A[bit_rev(a,bits)] = array[:]
    return A

# =============================================================================
# FFT: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================
def make_fix_twiddle(N,bits,fraction,offset=0.0, method="round"):
    twids = cfixpoint(bits,fraction,offset = offset, method = method)
    twids.from_complex(np.exp(-2*np.arange(N//2)*np.pi*1j/N))
    return twids

def iterffft_natural_DIT(DATA,twid,shiftreg,bits,fraction,staged,offset=0.0,method="round"):  #parse in data,tiddle factors (must be in bit reversed order for natural order in),
                                                                                 #how many bits fixpoint numbers are, fraction bits they are, offset, and rounding scheme.
    data=DATA.copy()
    N = data.data.size                                                           #how long is data stream
    stages = int(np.log2(N))
    if(len(shiftreg)!=stages and type(shiftreg) is not list):
        raise ValueError("shift register must be of type list, and its length must be that of log2(data length)")
        
    num_of_groups = 1                                                            #number of groups - how many subarrays are there?
    distance = N//2   
    if(staged is not None): bnd = staged
    else: bnd = N                                                           #how far between each fft arm?
    while num_of_groups < bnd:                                                     #basically iterates through stages
        for k in range(num_of_groups):                                           #iterate through each subarray
            jfirst = 2*k*distance                                          #index to beginning of a group
            jlast = jfirst + distance - 1                                  #first index plus offset - used to index whole group
            W=twid[k]

            slc1 = slice(jfirst,jlast+1,1)
            slc2 = slice(jfirst+distance, jlast+1+distance,1)
            tmp = (W * data[slc2]) >> bits - 1                         #slice off lower bit growth from multiply
            tmp.bits =bits                                                   #bits will = 2*bits+1 - hence - (bits+1)
            tmp.fraction=fraction                                            #fraction will = 2*(frac1+frac2) - hence - (bits-1)
            tmp.normalise()
            data[slc2] = data[slc1]-tmp
            data[slc1] = data[slc1]+tmp
        if shiftreg.pop():                                                       #implement FFT shift and then normalise to correct at end of stage
            data>>1
        data.normalise()
        
        num_of_groups *=2
        distance //=2
    if(staged == N or staged == None): 
        A=bitrevfixarray(data,N)                          #post bit-reordering
        return A
    else:
        return data

# =============================================================================
# Floating point PFB implementation making use of the natural order in fft
# like SARAO does. 
# =============================================================================
import matplotlib.pyplot as plt    

class FixPFB(object):
        """This function takes point size, how many taps, what percentage of total data to average over,
        what windowing function, whether you're running dual polarisations and staged (which stage) or not"""
        def __init__(self, N, taps, bits_in, bits_out, shiftreg, unsigned = False, offset = 0.0,method = "round",  avgperc = 1, datasrc = None, w = 'hanning',dual = False,staged = None):
            self.N = N                   #how many points
            self.avg = avgperc           #what averaging
            self.dual = dual             #whether you're performing dual polarisations or not
            self.taps = taps
            self.bits_in = bits_in
            self.bits_out = bits_out
            self.shiftreg = shiftreg
            self.unsigned = unsigned
            self.offset = offset
            self.method = method
            if (staged is not None):
                self.staged = 2**staged
            else:
                self.staged = staged
            
            self.reg_real = fixpoint(self.bits_in, self.bits_in,unsigned = self.unsigned,offset = self.offset, method = self.method)
            self.reg_real.from_float(np.zeros([N,taps],dtype = np.int64)) #our fir register size filled with zeros orignally
            self.reg_imag = self.reg_real.copy()
            
            self.inputdata = cfixpoint(self.bits_in, self.bits_in,unsigned = self.unsigned,offset = self.offset, method = self.method)
            self.inputdatadir = None
            if(datasrc is not None and type(datasrc)==str):             #if input data file is specified
                self.inputdatadir = datasrc
                self.outputdatadir = datasrc[:-4]+"out.npy"
                self.inputdata.from_complex(np.load(datasrc, mmap_mode = 'r'))
            
            WinDic = {                   #dictionary of various filter types
                'hanning' : np.hanning,
                'hamming' : np.hamming,
                'bartlett': np.bartlett,
                'blackman': np.blackman,
                }
            
            self.window = fixpoint(self.bits_out, self.bits_out-1,unsigned = self.unsigned,offset = self.offset, method = self.method)
            self.window.from_float(WinDic[w](taps))     
            self.X_k = None                 #our output
                
            self.twids = make_fix_twiddle(self.N,self.bits_out,self.bits_out-1,self.offset,self.method)
            self.twids = bitrevfixarray(self.twids,self.twids.data.size)
            
        """Takes data segment (N long) and appends each value to each fir.
        Returns data segment (N long) that is the sum of fircontents*window"""
        def _FIR(self,x):
            X_real = self.reg_real*self.window
            print(self.reg_real.bits)
            X_imag = self.reg_imag*self.window
            X = cfixpoint(real = X_real.sum(axis=1),imag = X_imag.sum(axis =1))
            print(X.bits)
            X >> (X.bits - self.bits_out)
            X.bits = self.bits_out
            X.fraction = self.bits_out
            X.normalise()
            self.reg_real.data = np.column_stack((x.real.data,self.reg_real.data))[:,:-1]
            self.reg_imag.data = np.column_stack((x.imag.data,self.reg_imag.data))[:,:-1]     #push and pop from FIR register array
            return X
        
        """In the event that that dual polarisations have been selected, we need to 
        split out the data after and return the individual X_k values"""
        def _split(self,Yk):            
            R_k = Yk.real.copy()
            I_k = Yk.imag.copy()
            
            R_kflip = R_k.copy()
            R_kflip[1:] = R_kflip[:0:-1]
            
            I_kflip = I_k.copy()
            I_kflip[1:] = I_kflip[:0:-1]

            self.G_k = cfixpoint(real = R_k + R_kflip, imag = I_k - I_kflip)
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
            if(self.avg ==1):
                retX = X.real*X.real + X.imag*X.imag
                return retX
            else:
                iterr = int(1/self.avg)
                rng = len(X.data[0,:])//iterr
                Xt = fixpoint(self.bits_out, self.bits_out,unsigned = self.unsigned,offset = self.offset, method = self.method)
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
            if (DATA is not None):                                  #if we are using an input data array
                if(self.bits_in != DATA.bits):
                    raise ValueError("Input data must match precision specified for input data")
                self.inputdata = DATA
            elif(self.inputdata is None):
                raise ValueError ("No input data for PFB specified.")

            size = self.inputdata.data.shape[0]                           #get length of data stream
            stages = size//self.N                               #how many cycles of commutator
            
            X = cfixpoint(self.bits_out, self.bits_out,unsigned = self.unsigned,offset = self.offset, method = self.method)
            X.from_complex(np.zeros([self.N,stages]))           #will be tapsize x stage
            
            for i in range(0,stages):                           #for each stage, populate all firs, and run FFT once
                if(i == 0):
                    X[:,i] = iterffft_natural_DIT(self._FIR(self.inputdata[i*self.N:i*self.N+self.N]),self.twids,self.shiftreg.copy(),self.bits_out,self.bits_out,self.staged)
                else:
                    X[:,i] = iterffft_natural_DIT(self._FIR(self.inputdata[i*self.N-1:i*self.N+self.N-1]),self.twids,self.shiftreg.copy(),self.bits_out,self.bits_out,self.staged)
            
            if(self.dual and self.staged is None): 
                self._split(X)
            elif(not self.dual and self.staged is None):
                self.X_k = X
            elif(self.dual and self.staged is not None):
                self.G_k = X
                self.H_k = X
            else:
                self.X_k = X
            
            if(self.inputdatadir is not None):             
                if(self.dual): 
                    np.save("pol_1_"+self.outputdatadir,self.G_k.to_float())          #save output data as complex (same pol ordering)
                    np.save("pol_2_"+self.outputdatadir,self.H_k.to_float())
                else:   
                    np.save(self.outputdatadir,self.X_k.to_float())

        """Plotting method to display the spectrum - has option to display input alongside"""
        def show(self,save=False,flnm = 'plot.png'):

            if(self.inputdatadir is not None):         #In the event we are writing to
                                                       #and reading from a file.
                if(self.dual):
                    gk = np.load("pol_1_"+self.outputdatadir,mmap_mode = 'r')
                    hk = np.load("pol_2_"+self.outputdatadir,mmap_mode = 'r')
                    fig = plt.figure(1)
                    plt.subplot(211)
                    plt.plot(gk)
                
                    plt.subplot(212)
                    plt.plot(hk)
                    if(save): fig.savefig(flnm)
                    plt.show()
                    
                else:
                    xk = np.load(self.outputdatadir,mmap_mode='r')
                    fig = plt.plot(xk)
                    if(save): fig.savefig(flnm)
                    plt.show()
            
            else:
                if(self.dual):
                    fig = plt.figure(1)
                    plt.subplot(211)
                    plt.plot(self.G_k.to_float())
                
                    plt.subplot(212)
                    plt.plot(self.H_k.to_float())
                    if(save): fig.savefig(flnm)
                    
                else:
                    fig = plt.plot(self.X_k.to_float())
                    if(save): fig.savefig(flnm)
                
