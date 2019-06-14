# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:45:20 2018

@author: talonmyburgh
"""

"""The hope here is to develop a fixed point array set to test a python-based 
PFB and compare it with one implemented with CASPER tools"""

##############################IMPORTS########################################
import numpy as np
import numba as nb
#############################################################################
class fixpoint(object):
    #takes number bits in full, number that is fractional part, minimum and 
    #maximum number representable, unsigned or signed integer, rounding or 
    #truncation
    def __init__(self,bits,fraction, min_int=None, max_int=None, unsigned=False, method = "ROUND"):
        self.method = method
        self.range = 2 ** bits #The range of the number (ie max)
        self.scale = 2 ** fraction #The decimal value length
        self.unsigned = unsigned
        self.__setbnds__(min_int,max_int)
        self.data = None
    
    def __setbnds__(self, min_int=None, max_int=None):
        if min_int is None: # decides minimal value
            self.min = 0 if self.unsigned else - self.range // 2
        else:
            self.min = min_int
        if max_int is None: #decides maximum value
            self.max = self.range - 1 if self.unsigned else self.range // 2 - 1
        else:
            self.max = max_int

    @property
    def bits(self): #define bits as a somewhat 'private' property
        return int(np.log2(self.range))
    
    @bits.setter
    def bits(self,val):
        if(type(val)!=int):
            raise ValueError("'bits' argument must be of type integer")
        else:
            self.range = 2 ** val
            self.__setbnds__()

    @property
    def fraction(self): #define frac as a somewhat 'private' property
        return int(np.log2(self.scale))
    
    @fraction.setter
    def fraction(self,val):
        if(type(val)!=int):
            raise ValueError("'frac' argument must be of type integer")
        else:
            self.scale = 2**val

    @property
    def unsigned(self): #define signed/unsigned as a somewhat 'private' property
        return self.min == 0
    
    @unsigned.setter
    def unsigned(self,val):
        if(type(val)!=bool):
            raise ValueError("'unsigned' argument must be of type bool")
        else:
            self.min = 0 if val else - self.range // 2
            self.max = self.range - 1 if val else self.range // 2 - 1
            
    @property
    def FPTYPE(self):
        if(self.unsigned):
            return np.uint64
        else:
            return np.int64
            
    def __repr__(self): #how things will be shown when using 'print'
        return 'FP real %s (%d, %d), shape %s' % \
               ('unsigned' if self.unsigned else 'signed',
                self.bits, self.fraction, np.shape(self.data))
               
    def __getitem__(self,key):
        newfpt = fixpoint(self.bits,self.fraction,unsigned=self.unsigned,method = self.method)
        newfpt.data = self.data.copy()[key]
        return newfpt
    
    def __setitem__(self,key,val):
        self.data[key] = val.data.copy()
    
    def normalise(self): #how to fit all data values within the range specified
        self.data = np.clip(self.data, self.min, self.max)

    def from_float(self, x): #take in float values
        if(self.method =="ROUND"): #if we're rounding off decimal values bankers style
            self.data = np.clip(np.round(x * self.scale).astype(self.FPTYPE),
                            self.min, self.max)
        elif(self.method =="TRUNCATE"): #if we're truncating off decimal
            self.data = np.clip(np.trunc(x * self.scale).astype(self.FPTYPE),
                            self.min, self.max)
        elif(self.method == "ROUND_INFTY"): #round to decimal as round up - much slower but only option now.
            self.data = np.clip(self.__roundinfty__(x * self.scale).astype(self.FPTYPE),
                               self.min,self.max)
        else:
            raise ValueError("No recognisable quantisation method specified")
        
    def to_float(self): #for plotting etc
        return (self.data.astype(self.FPTYPE)) / self.scale

    def sum(self, *args, **kwargs): #rewrite the sum method
        res = self.data.sum(*args, **kwargs) #use numpy sum method
        bits = self.bits + int(np.ceil(np.log2(self.data.size / res.size)))
        result = fixpoint(bits, self.fraction, unsigned=self.unsigned, method = self.method)
        result.data = res
        result.normalise() #clip and stuff
        return result
    
    def __mul__(self, w):
        res = self.data * w.data
        result = fixpoint(self.bits + w.bits,
                                 self.fraction + w.fraction,
                                 unsigned=self.unsigned and w.unsigned)
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
                                 unsigned=self.unsigned and y.unsigned) 
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
                                 unsigned=self.unsigned and y.unsigned)
        result.data = res
        result.normalise()
        return result
    
    def quantise(self, bits, fraction, min_int=None, max_int=None, unsigned=False,method="ROUND"):
        result = fixpoint(bits, fraction, min_int, max_int, unsigned)
        result.from_float(self.to_float())
        return result
    
    def __rshift__(self,steps):#slicing and right shifting technique - allows for rounding
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
    
    def copy(self):  #method for making a copy of fixpoint type (else get referencing issues)
        tmpfxpt=fixpoint(self.bits,self.fraction,unsigned=self.unsigned,method = self.method)
        tmpfxpt.data = self.data.copy()
        return tmpfxpt
    
    @nb.jit
    def __roundinfty__(self,array):
        a = array.copy()
        f=np.modf(a)[0]             #get decimal values from data
        if (a.ndim == 1):
            for i in range(len(array)):
                if((f[i]<0.0 and f[i] <=-0.5) or (f[i]>=0.0 and f[i]<0.5)):
                    a[i]=np.floor(a[i])
                else:
                    a[i]=np.ceil(a[i])
        elif(a.ndim==2):
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if((f[i,j]<0.0 and f[i,j] <=-0.5) or (f[i,j]>=0.0 and f[i,j]<0.5)):
                        a[i,j]=np.floor(a[i,j])
                    else:
                        a[i,j]=np.ceil(a[i,j])
        elif(a.ndim==3):
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        if((f[i,j,k]<0.0 and f[i,j,k] <=-0.5) or (f[i,j,k]>=0.0 and f[i,j,k]<0.5)):
                            a[i,j,k]=np.floor(a[i,j,k])
                        else:
                            a[i,j,k]=np.ceil(a[i,j,k])
        return a
    
    __str__ = __repr__

class cfixpoint(object):
    """Fixed-point container for complex values."""
    def __init__(self, bits=None, fraction=None, min_int=None, max_int=None, unsigned=False,
                 method = "ROUND",real=None, imag=None):

        if bits is not None: #basically if bits and isn't None
            self.real = fixpoint(bits, fraction, min_int, max_int, unsigned, method)
            self.imag = fixpoint(bits, fraction, min_int, max_int, unsigned, method)
        elif real is not None:
            self.real = real
            self.imag = imag
        else:
            raise ValueError("Must either specify bits/fraction or pass two fixpoint numbers to real/imag.")

    @property
    def bits(self):
        return int(np.log2(self.real.range))
    
    @bits.setter
    def bits(self,val):
        self.real.bits = val
        self.imag.bits = val

    @property
    def fraction(self):
        return int(np.log2(self.real.scale))
    
    @fraction.setter
    def fraction(self,val):
        self.real.fraction = val
        self.imag.fraction = val

    @property
    def range(self):
        return self.real.range

    @property
    def scale(self):
        return self.real.scale

    @property
    def unsigned(self):
        return self.real.min == 0
    
    @unsigned.setter
    def unsigned(self,val):
        self.real.unsigned=val
        self.imag.unsigned=val

    @property
    def min(self):
        return self.real.min + 1j * self.imag.min

    @property
    def max(self):
        return self.real.max+ 1j * self.imag.max

    @property
    def data(self):
        return self.real.data + 1j * self.imag.data  
    
    @property
    def method(self):
        return self.real.method
    
    @method.setter
    def method(self,val):
        self.real.method=val
        self.imag.method=val

    def __repr__(self):
        return 'FP complex %s (%d, %d), shape %s' % \
               ('unsigned' if self.unsigned else 'signed',
                self.bits, self.fraction, np.shape(self.real.data))
    
    def __getitem__(self,key):
        tmpcfpt = cfixpoint(real = self.real[key],imag = self.imag[key])
        return tmpcfpt
    
    def __setitem__(self,key,val):
        self.real[key] = val.real
        self.imag[key] = val.imag


    def from_complex(self, x):
        self.real.from_float(x.real)
        self.imag.from_float(x.imag)

    def to_complex(self):
        return self.real.to_float() + 1j * self.imag.to_float()

    def counts(self, select=None):
        return self.real.counts(select) + self.imag.counts(select)

    def sum(self, *args, **kwargs):
        result = cfixpoint(real=self.real.sum(*args, **kwargs),
                                        imag=self.imag.sum(*args, **kwargs))
        return result

    def __mul__(self, w):
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

    def __add__(self, y):
        result = cfixpoint(real=self.real+y.real,
                                        imag=self.imag+y.imag)
        return result

    def __sub__(self, y):
        result = cfixpoint(real=self.real-y.real,
                                        imag=self.imag-y.imag)
        return result
    @staticmethod
    def concatenate(other,another):
        tmpcfpt = cfixpoint(other.bits,other.fraction,unsigned = other.unsigned)
        tmpcfpt.real =fixpoint.concatenate(other.real,another.real)
        tmpcfpt.imag =fixpoint.concatenate(other.imag,another.imag)
        return tmpcfpt
    
    def normalise(self):
        self.real.normalise()
        self.imag.normalise()

    def quantise(self, bits, fraction, min_int=None, max_int=None, unsigned=False):
        out_real = self.real.quantise(bits, fraction, min_int, max_int, unsigned)
        out_imag = self.imag.quantise(bits, fraction, min_int, max_int, unsigned)
        result = cfixpoint(real=out_real, imag=out_imag)
        return result
    
    def __rshift__(self,steps):
        self.real >> steps
        self.imag >> steps
        return self
    
    def __lshift__(self,steps):
        self.real << steps
        self.imag << steps
        return self
    
    def copy(self): #method for making a copy of fixpoint type (else get referencing issues)
        tmpcfxpt = cfixpoint(real = self.real.copy(),imag=self.imag.copy())
        return tmpcfxpt
    
    __str__ = __repr__
    