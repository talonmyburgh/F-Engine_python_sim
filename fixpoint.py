# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:45:20 2018

@author: Talon Myburgh
"""

"""The hope here is to develope a fixed point array set to test a python-based 
PFB and compare it with one implemented with CASPER tools"""

##############################IMPORTS########################################
import numpy as np
from collections import Counter
#############################################################################
class fixpoint(object):
    #takes number bits in full, number that is fractional part, minimum and 
    #maximum number representable, unsigned or signed integer, rounding or 
    #truncation and offset val
    def __init__(self,bits,fraction,min_int=None, max_int=None, unsigned=False, offset=0.0, method = "round",FPTYPE = np.int64):
        
        self.FPTYPE = FPTYPE
        self.method = method
        self.range = 2 ** bits #The range of the number (ie max)
        self.scale = 2 ** fraction #The decimal value length
        self.unsigned = unsigned
        self.__setbnds(min_int,max_int)
        self.data = None
        self.offset = offset
    
    def __setbnds(self, min_int=None, max_int=None):
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
            self.__setbnds()

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
    
    def __repr__(self): #how things will be shown when using 'print'
        return 'FP real %s (%d, %d), shape %s' % \
               ('unsigned' if self.unsigned else 'signed',
                self.bits, self.fraction, np.shape(self.data))
               
    def __getitem__(self,key):
        newfpt = self.copy()
        newfpt.data = self.data.copy()[key]
        return newfpt
    
    def __setitem__(self,key,val):
        self.data[key] = val.data
    
    def normalise(self): #how to fit all data values within the range specified
        self.data = np.clip(self.data, self.min, self.max)

    def from_float(self, x): #take in float values
        if(self.method =="round"): #if we're rounding off decimal values
            self.data = np.clip(np.round(x * self.scale + self.offset).astype(self.FPTYPE),
                            self.min, self.max)
        elif(self.method =="truncate"): #if we're truncating off decimal
            self.data = np.clip(np.trunc(x * self.scale + self.offset).astype(self.FPTYPE),
                            self.min, self.max)
            
    def to_float(self): #for plotting etc
        return (self.data.astype(np.float) - self.offset) / self.scale
    
    def counts(self, select=None): #pulls out a set number of values from the start of data array
                                    #and counts recurring instances of the value
        select = slice(None) if select is None else select
        return Counter(self.data[select].flat)

    def sum(self, *args, **kwargs): #rewrite the sum method
        res = self.data.sum(*args, **kwargs) #use numpy sum method
        bits = self.bits + int(np.ceil(np.log2(self.data.size / res.size)))
        result = fixpoint(bits, self.fraction, unsigned=self.unsigned, offset = self.offset, method = self.method)
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
        if(self.scale>y.scale or self.scale<y.scale):
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
 
    @staticmethod
    def concatenate(other,another):
        tmparray = np.concatenate([other.data,another.data])
        tmpfpt = fixpoint(other.bits,other.fraction,unsigned = other.unsigned)
        tmpfpt.data = tmparray
        return tmpfpt
    
    def quantise(self, bits, fraction, min_int=None, max_int=None, unsigned=False):
        result = fixpoint(bits, fraction, min_int, max_int, unsigned)
        result.from_float(self.to_float())
        return result
    
    def __rshift__(self,steps):
        self.data >>= steps
        return self
    
    def __lshift__(self,steps):
        self.data <<= steps
        return self
    
    def copy(self):  #method for making a copy of fixpoint type (else get referencing issues)
        tmpfxpt=fixpoint(self.bits,self.fraction,unsigned=self.unsigned,offset=self.offset,method = self.method)
        tmpfxpt.data = self.data.copy()
        return tmpfxpt
    
    __str__ = __repr__

class cfixpoint(object):
    """Fixed-point container for complex values."""
    def __init__(self, bits=None, fraction=None, min_int=None, max_int=None, unsigned=False,
                 offset = 0.0,method = "round",real=None, imag=None):
        
        self.offset = offset
        self.method = method
        
        if bits: #basically if bits and isn't None
            self.real = fixpoint(bits, fraction, min_int, max_int, unsigned,offset,method)
            self.imag = fixpoint(bits, fraction, min_int, max_int, unsigned,offset,method)
        else:
            self.real = real
            self.imag = imag

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

    @property
    def min(self):
        return self.real.min + 1j * self.imag.min

    @property
    def max(self):
        return self.real.max+ 1j * self.imag.max

    @property
    def data(self):
        return self.real.data + 1j * self.imag.data  

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
        
    def complex_randn(*args):
        return np.random.randn(*args) + 1.0j * np.random.randn(*args)
    
    def copy(self): #method for making a copy of fixpoint type (else get referencing issues)
        tmpcfxpt = cfixpoint(real = self.real.copy(),imag=self.imag.copy())
        return tmpcfxpt
    
    __str__ = __repr__
    
    
