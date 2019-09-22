# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:45:20 2018

@author: talonmyburgh
"""

"""The hope here is to develop a fixed-point array set to test a python-based 
PFB and compare it with one implemented using the CASPER toolset"""

##############################IMPORTS########################################
import numpy as np
import numba as nb
#############################################################################

class fixpoint(object):
    """Takes number bits in full, number of fractional bits, minimum and 
    maximum number representable, unsigned or signed integer, rounding method
    and overflow method"""
    def __init__(self,bits,fraction, min_int=None, max_int=None,
                 unsigned=False, method = "ROUND"):
        self.method = method
        self.range = 2 ** bits                                                 #The dynamic range of the number
        self.scale = 2 ** fraction                                             #The fractional dynamic range by which the number will be scaled
        self.unsigned = unsigned
        self.__setbnds__(min_int,max_int)                                      #Sets self.min and self.max of number
        self.data = None
    
    def __setbnds__(self, min_int=None, max_int=None):
        if min_int is None:                                                    # decides minimal value
            self.min = 0 if self.unsigned else - self.range // 2
        else:
            self.min = min_int
        if max_int is None:                                                    #decides maximum value
            self.max = self.range - 1 if self.unsigned else self.range // 2 - 1
        else:
            self.max = max_int

    @property
    def bits(self):                                                            #bits property
        return int(np.log2(self.range))
    
    @bits.setter
    def bits(self,val):
        if(type(val)!=int):
            raise ValueError("'bits' argument must be of type integer")
        else:
            self.range = 2 ** val
            self.__setbnds__()

    @property
    def fraction(self):                                                        #frac property
        return int(np.log2(self.scale))
    
    @fraction.setter
    def fraction(self,val):
        if(type(val)!=int):
            raise ValueError("'frac' argument must be of type integer")
        else:
            self.scale = 2**val

    @property
    def unsigned(self):                                                        #unsigned as property
        return self.min == 0
    
    @unsigned.setter
    def unsigned(self,val):
        if(type(val)!=bool):
            raise ValueError("'unsigned' argument must be of type bool")
        else:
            self.min = 0 if val else - self.range // 2
            self.max = self.range - 1 if val else self.range // 2 - 1
            
    @property                                                                  #64bit int for signed and 64bit uint for unsigned 
    def FPTYPE(self):
        if(self.unsigned):
            return np.uint64
        else:
            return np.int64
            
    def __repr__(self):                                                        #how things will be shown when using 'print'
        return 'FP real %s (%d, %d), shape %s' % \
               ('unsigned' if self.unsigned else 'signed',
                self.bits, self.fraction, np.shape(self.data))
               
    def __getitem__(self,key):                                                 #method of slicing fixpoint arrays
        newfpt = fixpoint(self.bits,self.fraction,unsigned=self.unsigned,
                          method = self.method)
        newfpt.data = self.data.copy()[key]
        return newfpt
    
    def __setitem__(self,key,val):                                             #method for populating slices of arrays
        self.data[key] = val.data.copy()
    
    def normalise(self):                                                       #how to fit all data values within the min/max specified
        self.data = np.clip(self.data, self.min, self.max)

    def from_float(self, x):                                                   #take in float values                                       #detect overflow method used
        if(self.method =="ROUND"):                                             #if we're rounding off decimal values bankers style
            self.data = np.clip(np.round(x*self.scale).astype(self.FPTYPE),
                            self.min, self.max)
        elif(self.method =="TRUNCATE"):                                        #if we're truncating off decimal
            self.data = np.clip(np.trunc(x*self.scale).astype(self.FPTYPE),
                            self.min, self.max)
        elif(self.method == "ROUND_INFTY"):                                    #round to decimal as round up - much slower but only option now.
            self.data = np.clip(self.__roundinfty__(x*self.scale).astype(self.FPTYPE),
                               self.min,self.max)
        else:
            raise ValueError("No recognisable quantisation method specified")
        
    def to_float(self): #for plotting etc
        return (self.data.astype(self.FPTYPE)) / self.scale

    def sum(self, *args, **kwargs):                                            #rewrite the sum method
        res = self.data.sum(*args, **kwargs)                                   #use numpy sum method
        bits = self.bits + int(np.ceil(np.log2(self.data.size / res.size)))
        result = fixpoint(bits, self.fraction, unsigned=self.unsigned,
                          method = self.method)
        result.data = res
        result.normalise()                                                     #clip and stuff
        return result
    
    def __mul__(self, w):
        res = self.data * w.data
        result = fixpoint(self.bits + w.bits,
                                 self.fraction + w.fraction,
                                 unsigned=self.unsigned and w.unsigned,
                                 method = self.method)
        result.data = res
        result.normalise()
        return result

    def __add__(self, y):
        if(self.scale!=y.scale):
            raise ValueError("Addition performed between two numbers of differing scales!")
            
        res = self.data + y.data
        #adds together, and accounts for carry bit
        result = fixpoint(max(self.bits, y.bits) + 1,
                                 max(self.fraction, y.fraction),
                                 unsigned=self.unsigned and y.unsigned,
                                 method = self.method) 
        result.data = res
        result.normalise()
        return result

    def __sub__(self, y):
        if(self.scale>y.scale or self.scale<y.scale):
            raise ValueError("Subtraction performed between two numbers of differing scales!")
            
        res = self.data - y.data
        #subtracts together, and accounts for carry bit
        result = fixpoint(max(self.bits, y.bits) + 1,
                                 max(self.fraction, y.fraction),
                                 unsigned=self.unsigned and y.unsigned,
                                 method = self.method)
        result.data = res
        result.normalise()
        return result
    
    def quantise(self, bits, fraction, min_int=None, max_int=None, unsigned=False,
                 method="ROUND"):
        result = fixpoint(bits, fraction, min_int, max_int, unsigned,
                          method = method)
        result.from_float(self.to_float())
        return result
    
    def __rshift__(self,steps):                                                #slicing and right shifting technique - allows for rounding
        if(self.method == "ROUND"):
            self.data = np.round(self.data/(2**steps)).astype(self.FPTYPE)
        elif(self.method =="ROUND_INFTY"):
            self.data = self.__roundinfty__(self.data/(2**steps)).astype(self.FPTYPE)
        elif(self.method=="TRUNCATE"):    
            self.data >>= steps
        else:
            raise ValueError("No recognisable quantisation method specified")
        return self
    
    def __lshift__(self,steps):
        self.data <<= steps
        return self
    
    def copy(self):                                                            #method for making a copy of fixpoint type (else get referencing issues)
        tmpfxpt=fixpoint(self.bits,self.fraction,unsigned=self.unsigned,
                         method = self.method,
                         min_int = self.min, max_int = self.max)
        tmpfxpt.data = self.data.copy()
        return tmpfxpt
    
    def power(self):
        return self.data*self.data
    
    """This method rounds values in an array to +/- infinity"""
    @nb.jit(nopython = True)
    def __roundinfty__(self,array):
        a = array.copy()
        f=np.modf(a)[0]                                                        #get decimal values from data
        if (a.ndim == 1):                                                      #for 1D array
            for i in range(len(array)):
                if((f[i]<0.0 and f[i] <=-0.5) or (f[i]>=0.0 and f[i]<0.5)):
                    a[i]=np.floor(a[i])
                else:
                    a[i]=np.ceil(a[i])
        elif(a.ndim==2):                                                       #for 2D array
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if((f[i,j]<0.0 and f[i,j] <=-0.5) or (f[i,j]>=0.0
                       and f[i,j]<0.5)):
                        a[i,j]=np.floor(a[i,j])
                    else:
                        a[i,j]=np.ceil(a[i,j])
        elif(a.ndim==3):                                                       #for 3D array
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        if((f[i,j,k]<0.0 and f[i,j,k] <=-0.5) or 
                           (f[i,j,k]>=0.0 and f[i,j,k]<0.5)):
                            a[i,j,k]=np.floor(a[i,j,k])
                        else:
                            a[i,j,k]=np.ceil(a[i,j,k])
        return a

    __str__ = __repr__                                                         #redundancy for print
    
    """Fixed-point container for complex values which makes use of existing 
    fixpoint. Additional parameters here are to specify two fixpoint numbers as
    real and imag, by which cfixpoint will extract all other parameters."""
class cfixpoint(object):

    def __init__(self, bits=None, fraction=None, min_int=None, max_int=None,
                 unsigned=False, method = "ROUND", real=None, imag=None):

        if bits is not None:                                                   #if bits are supplied (i.e not real and imag)
            self.real = fixpoint(bits, fraction, min_int, max_int, unsigned,   #declare a real and imag fixpoint
                                 method)
            self.imag = fixpoint(bits, fraction, min_int, max_int, unsigned,
                                 method)
        elif real is not None:                                                 #else use real and imag fixpoint supplied
            self.real = real
            self.imag = imag
        else:
            raise ValueError("Must either specify bits/fraction or pass two fixpoint numbers to real/imag.")

    @property                                                                  #bits property
    def bits(self):
        return int(np.log2(self.real.range))
    
    @bits.setter
    def bits(self,val):
        self.real.bits = val
        self.imag.bits = val

    @property                                                                  #fraction property
    def fraction(self):
        return int(np.log2(self.real.scale))
    
    @fraction.setter
    def fraction(self,val):
        self.real.fraction = val
        self.imag.fraction = val

    @property                                                                  #range
    def range(self):
        return self.real.range

    @property                                                                  #scale
    def scale(self):
        return self.real.scale

    @property                                                                  #unsigned property
    def unsigned(self):
        return self.real.min == 0
    
    @unsigned.setter
    def unsigned(self,val):
        self.real.unsigned=val
        self.imag.unsigned=val

    @property                                                                  #min property
    def min(self):
        return self.real.min + 1j * self.imag.min

    @property
    def max(self):                                                             #max property
        return self.real.max+ 1j * self.imag.max

    @property
    def data(self):                                                            #data held in cfixpoint (will be integer)
        return self.real.data + 1j * self.imag.data  
    
    @property
    def method(self):                                                          #rounding method in use
        return self.real.method
    
    @method.setter
    def method(self,val):
        self.real.method=val
        self.imag.method=val
    
    def __repr__(self):                                                        #printing 
        return 'FP complex %s (%d, %d), shape %s' % \
               ('unsigned' if self.unsigned else 'signed',
                self.bits, self.fraction, np.shape(self.real.data))
    
    def __getitem__(self,key):                                                 #returning slices of array
        tmpcfpt = cfixpoint(real = self.real[key],imag = self.imag[key])
        return tmpcfpt
    
    def __setitem__(self,key,val):                                             #setting slices of array
        self.real[key] = val.real
        self.imag[key] = val.imag


    def from_complex(self, x):                                                 #accepts complex array and populates to data with scaling
        self.real.from_float(x.real)
        self.imag.from_float(x.imag)

    def to_complex(self):                                                      #converts data to complex array
        return self.real.to_float() + 1j * self.imag.to_float()

    def sum(self, *args, **kwargs):
        result = cfixpoint(real=self.real.sum(*args, **kwargs),
                                        imag=self.imag.sum(*args, **kwargs))
        return result

    def __mul__(self, w):                                                      #complex multiplication
        def complex_mult(a, b, c, d):
            """Returns complex product x + jy = (a + jb) * (c + jd)."""
            # Real part x = a*c - b*d
            x = (a*c)-(b*d)
            # Imaginary part y = a*d + b*c
            y = (a*d)+(b*c)
            return x, y
        out_real, out_imag = complex_mult(self.real, self.imag, w.real, w.imag)
        result = cfixpoint(real=out_real, imag=out_imag)
        return result

    def __add__(self, y):                                                      #complex addition
        result = cfixpoint(real=self.real+y.real,
                                        imag=self.imag+y.imag)
        return result

    def __sub__(self, y):                                                      #complex subtraction
        result = cfixpoint(real=self.real-y.real,
                                        imag=self.imag-y.imag)
        return result
    
    def normalise(self):                                                       #normalise the real and imag data
        self.real.normalise()
        self.imag.normalise()

    #quantise the data to bounds required.
    def quantise(self, bits, fraction, min_int=None, max_int=None,
                 unsigned=False, method="ROUND"):
        out_real = self.real.quantise(bits, fraction, min_int, max_int,
                                      unsigned, method)
        out_imag = self.imag.quantise(bits, fraction, min_int, max_int,
                                      unsigned, method)
        result = cfixpoint(real=out_real, imag=out_imag)
        return result
    
    def __rshift__(self,steps):                                                #right shift data by steps
        self.real >> steps
        self.imag >> steps
        return self
    
    def __lshift__(self,steps):                                                #left shift data by steps
        self.real << steps
        self.imag << steps
        return self
    
    def copy(self):                                                            #method for making a copy of cfixpoint type
        tmpcfxpt = cfixpoint(real = self.real.copy(),imag=self.imag.copy())
        return tmpcfxpt
    
    def conj(self):                                                            #returns conjugate of cfixpoint
        i_res = self.imag.copy()
        i_res.data = -self.imag.data.copy()
        res = cfixpoint(real=self.real,imag=i_res)
        return res
    
    def power(self):                                                           #return power as a x a* of cfixpoint
        res = self.copy() * self.conj() 
        return res.real
        
    __str__ = __repr__
    