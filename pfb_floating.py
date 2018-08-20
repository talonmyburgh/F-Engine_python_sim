# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:21:43 2018

@author: User
"""
import numpy as np

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

def bitrevarray(array,N):
    bits = int(np.log2(N))
    A = np.zeros(N,dtype=np.complex)
    for k in range(0, N):
        A[bit_rev(k,bits)] = array[k]
    return A

# =============================================================================
# FFT 1: bit reverse order in, natural order out.
# =============================================================================

def iterfft_bitorder_in_DIT(a):
    N=len(a)                                   #how long is data stream
    A=bitrevarray(a,N)                         #bit reorder input data prior to processing
    for stage in range(1,int(np.log2(N))+1):   #iterate through stages
        subarraylen = 2**stage                 #size of sub-array being processed - will decrease if natural order in, else it increases 
                                               #(i.e. prior decimation vs post decimation).
        w_m = np.exp(-2*np.pi*1j/subarraylen)  #twiddle factor base
        distance = int(subarraylen/2)          #distance between input arms to butterflies
        for k in range(0,N,subarraylen):       #iterate through all sub arrays
            w=1                                #start at 1
            for j in range(0,distance):        #iterate through each pair of arms in butterflies
                t= w*A[k+j+distance]           #one butterfly input arm
                u = A[k+j]                     #other butterfly input arm
                A[k+j]=u+t                     #one butterfly output arm
                A[k+j+distance]=u-t            #other butterfly output arm
                w=w*w_m                        #update twiddle factor - rotate it for next iteration
    return A

# =============================================================================
# FFT 2: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================

def make_twiddle(N):                             #generate array of needed twiddle
    arr = np.zeros(N//2,dtype=np.complex)
    for i in range(N//2):
        arr[i] = np.exp(-2*i*np.pi*1j/N)
    return arr

def iterfft_natural_in_DIT(s,w):
    a = np.asarray(s,dtype = np.complex)
    N = a.size                                  #how long is data stream
    pairs_in_group = N//2                       #how many butterfly pairs per group - starts at 1/2*full data length obviously
    num_of_groups = 1                           #number of groups - how many subarrays are there?
    distance = N//2                             #how far between each fft arm?
    while num_of_groups < N:                    #basically iterates through stages
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
    A=bitrevarray(a,N)                          #post bit-reordering
    return A


# =============================================================================
# Floating point PFB implementation making use of the natural order in fft
# like SARAO does. 
# =============================================================================

import matplotlib.pyplot as plt

class PFB(object):
        """This function takes point size, how many taps, what percentage of total data to average over,
        what windowing function, whether you're running dual polarisations, and data type"""
        def __init__(self, N, taps, avg = 1, datasrc = None, w = 'hanning',dual = False):
            self.N = N                                      #how many points
            self.avg = avg                                  #what averaging
            self.dual = dual                                #whether you're performing dual polarisations or not
            self.reg =np.zeros([N,taps])                    #our fir register size filled with zeros orignally
            self.inputdata = None
            if(datasrc is not None and type(datasrc)==str):             #if input data file is specified
                self.inputdatadir = datasrc
                self.outputdatadir = datasrc[:-4]+"out.npy"
                self.inputdata  = np.load(datasrc)
                
            WinDic = {                                      #dictionary of various filter types
                'hanning' : np.hanning,
                'hamming' : np.hamming,
                'bartlett': np.bartlett,
                'blackman': np.blackman,
                }
            
            self.window=WinDic[w](taps)     
            self.X_k = None                                 #our output
            
            if(dual):                                       #if dual, our two outputs
                self.H_k = None
                self.G_k = None
                
            self.twids = maketwiddle(self.N)
                
        """Takes data segment (N long) and appends each value to each fir.
        Returns data segment (N long) that is the sum of fircontents*window"""
        def _FIR(self,x):
            X = np.sum(self.reg*self.window,axis=1)              #filter
            self.reg = np.column_stack((x,self.reg))[:,:-1]      #push and pop from FIR register array
            return X
        
        """In the event that that dual polarisations have been selected, we need to 
        split out the data after and return the individual X_k values"""        
        def _split(self,Y_k):
            length = np.size(Y_k,axis=1)
            self.H_k = np.zeros([self.N,length],dtype = np.complex)
            self.G_k = np.ones([self.N,length],dtype=np.complex)
            
            #reverse the arrays for the splitting function correctly
            R_k = np.real(Y_k)
            R_kflip = R_k.copy()
            R_kflip[1:] = R_kflip[1:][::-1]
            
            I_k = np.imag(Y_k)
            I_kflip = I_k.copy()
            I_kflip[1:] = I_kflip[1:][::-1]

            self.G_k[:,:] = (1/2)*(R_k[:,:]+1j*I_k[:,:]+R_kflip[:,:]-1j*I_kflip[:,:])
            self.H_k[:,:] = (1/2j)*(R_k[:,:]+1j*I_k[:,:]-R_kflip[:,:]+1j*I_kflip[:,:])
        
        

        """Here we take the power spectrum of the outputs. The averaging scheme
        tells over what portion of the output data to take the power spectrum of.
        If alternatively a input file is specified, then average specifies over
        how many channels to average."""        
        def _pow(self,X):
            if(self.avg ==1):
                return np.abs(X[:])**2                             #The scheme for averaging used.
            
            elif(abs(self.avg)<1):                                 #If input data array is used
                                                                   #then averaging will be over a % of it.
                iterr = int(1/self.avg)
                rng = X.shape[1]//iterr
                Xt = np.zeros([self.N,iterr])
                for i in range(0,iterr):
                    if(i ==0):
                        Xt[:,i] = np.abs(np.sum(X[:,0:rng],axis=1))**2
                    else:
                        Xt[:,i] = np.abs(np.sum(X[:,i*rng-1:i*rng+rng-1],axis=1))**2
                return Xt
            
            else:                                                   #If input file is used the averaging then
                                                                    #averaging will be over specified # of channels.
                assert X.shape[1]>self.avg and X.shape[1]%self.avg==0, "Data parsed is not enough to average over"
                +"or is not multiple of average specified"
                iterr = X.shape[1]//self.avg
                Xt = np.zeros([self.N,iterr])
                for i in range(0,iterr):
                    if(i==0):
                        Xt[:,i] = np.abs(np.sum(X[:,0:self.avg],axis=1))**2
                    else:
                        Xt[:,i] = np.abs(np.sum(X[:,i*self.avg-1:i*self.avg+self.avg-1],axis=1))**2

        """Given data, (having specified whether the PFB will run in dual or not)
        you parse the data and the PFB will compute the spectrum (continuous data mode to still add)"""
        def run(self,data=None):
            if (data is not None):                                  #if we are using an input data array
                self.inputdata = data
            elif(self.inputdata is None):
                raise ValueError ("No input data for PFB specified.")

            size = len(self.inputdata)                                    #get length of data stream
            stages = size//self.N                               #how many cycles of commutator
            X = np.zeros([self.N,stages],dtype = np.complex)    #will be tapsize x stage
            
            for i in range(0,stages):                           #for each stage, populate all firs, and run FFT once
                if(i ==0):
                    X[:,i] = np.fft.fft(self._FIR(self.inputdata[i*self.N:i*self.N+self.N]))
                else:
                    X[:,i] = np.fft.fft(self._FIR(self.inputdata[i*self.N-1:i*self.N+self.N-1]))
            
            if(type(self.inputdata) == str):             
                if(self.dual): 
                    self._split(X)
                    np.save(self.outputdatadir,self._pow(self.G_k)+1j*self._pow(self.H_k))          #save output data as complex (same pol ordering)
                    
                else:   
                    np.save(self.outputdatadir,self._pow(X))
            else:
                if(self.dual): 
                    self._split(X)
                    self.G_k=self._pow(self.G_k)
                    self.H_k=self._pow(self.H_k)
                else:
                    self.X_k = self._pow(X)
                    

        """Plotting method to display the spectrum - has option to display input alongside"""
        def show(self,save=False,flnm = 'plot.png'):
            n = np.arange(self.N)
            if(type(self.inputdata) == str):         #In the event we are writing to
                                                    #and reading from a file.
                if(self.dual):
                    Val = np.load(self.outputdatadir)
                    fig = plt.figure(1)
                    plt.subplot(211)
                    plt.plot(n,np.real(Val))
                
                    plt.subplot(212)
                    plt.plot(n,np.imag(Val))
                    if(save): fig.savefig(flnm)
                    plt.show()
                    
                else:
                    Val = np.load(self.outputdatadir)
                    fig = plt.plot(n,Val)
                    if(save): fig.savefig(flnm)
                    plt.show()
            else:                                   #In the event we were given
                                                    #a direct array.
                if(self.dual):
                    fig = plt.figure(1)
                    plt.subplot(211)
                    plt.plot(n,self.H_k)
                
                    plt.subplot(212)
                    plt.plot(n,self.G_k)
                    if(save): fig.savefig(flnm)
                    plt.show()
                    
                else:
                    fig = plt.plot(n,self.X_k)
                    if(save): fig.savefig(flnm)
                    plt.show()