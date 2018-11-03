# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:23:55 2018

@author: User
"""

import numpy as np
from pfb_fixed import FixPFB
from pfb_floating import FloatPFB
from fixpoint import cfixpoint
import matplotlib.pyplot as plt

bits_out =17
bits_in = 9
method = "round"
taps = 16
point = 2048 #2k
N=2**16
n=np.arange(N)

####SIGNALS######
sig1 = np.cos(2**10*np.pi*n/N)
sig2 = np.cos(2**12*np.pi*n/N)
#sig2[80:100]=0.2
#sig3 = 1j*sig2+sig1
#realmeerkat = np.load('noplanes.npy',mmap_mode = 'r')
#sampmeerkat = realmeerkat[4096:12288] #8k
#sampmeerkat = sampmeerkat/100
#rnd1 = (np.random.random(N)-0.5)/2.5
#rnd2 = (np.random.random(N)-0.5)/2.5
#rnd = rnd1+1j*rnd2
#cs1 = np.cos(983.04*np.pi*(n/N),dtype = np.float64)/2.5
#cs2 = np.cos(1966.08*np.pi*(n/N),dtype = np.float64)/2.5
twotone = (sig1+sig2)/2.2

fsig = cfixpoint(bits_in,bits_in,method = method)
#fsig2 = cfixpoint(bits,fraction,method = method)
#fsig3 = cfixpoint(bits,fraction,method = method)
#frnd = cfixpoint(bits,fraction,method = method)
fsig.from_complex(twotone)
#fsig2.from_complex(sig2)
#fsig3.from_complex(sig3)
#frnd.from_complex(rnd)
#fsampmeerkat = cfixpoint(bits,fraction,method = method)
#fsampmeerkat.from_complex(sampmeerkat)

#s = np.arange(2.5,12.5,1)
#x = np.arange(point)
#outputpowfloat = np.zeros(20)
#outputpowfix = outputpowfloat.copy()
#inputpow = np.zeros(20)
#shiftreg = [0,0,0,0,0,0,0,0,0,0,0]
shiftreg = [1,1,1,1,1,1,1,1,1,1,1]
#shiftreg = [0,1,0,1,0,1,0,1,0,1]
#shiftreg = [0,0,0,0,0,1,1,1,1,1]
#shiftreg = [1,1,1,1,1,0,0,0,0,0]

#pfb_floating_single = FloatPFB(point,taps)
#pfb_floating_single.run(twotone)
#np.save("./F_Enginesim_float_twotone_output",pfb_floating_single.X_k[:,-1])

pfb_fixed_single = FixPFB(point,taps,bits_in,bits_out,shiftreg = shiftreg,method=method)
pfb_fixed_single.run(fsig)
np.save("./F_Enginesim_fixed_out_twotone_oned_FIRchange",pfb_fixed_single.X_k.to_complex()[:,-1])

fpgadatazero = np.load('raw_data_zerod.npy')
fpgadataoned = np.load('raw_data_oned.npy')
simdatazero = np.load('F_Enginesim_fixed_out_twotone_zerod_FIRchange')
simdataoned = np.load('F_Enginesim_fixed_out_twotone_oned_FIRchange')

plt.figure(1)
plt.subplot(211)
plt.plot(np.imag(simdatazero)[:1024], 'k',label = 'simulator f-engine data')
plt.subplot(212)
plt.plot(np.imag(fpgadatazero), 'k', label = 'fpga f-engine data')
plt.title('FPGA vs Simulator Imaginary data for zerod shift register')
plt.xlabel('bins')
plt.ylabel('$Im\{output\}$')
plt.legend()
plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(np.imag(simdataoned)[:1024], 'k',label = 'simulator f-engine data')
plt.subplot(212)
plt.plot(np.imag(fpgadataoned), 'k', label = 'fpga f-engine data')
plt.title('FPGA vs Simulator Real data for zerod shift register')
plt.xlabel('bins')
plt.ylabel('$Re\{output\}$')
plt.legend()
plt.show()

#while st < 11:
#    print(st)
#    cs1 = np.cos(983.04*np.pi*(n/N),dtype = np.float64)/s[st-1]
#    cs2 = np.cos(1966.08*np.pi*(n/N),dtype = np.float64)/s[st-1]
#    twotone = cs1+cs2
#    fsig1.from_complex(twotone)
#    inputpow[st-1] = np.sum(np.abs(twotone)**2)
#    pfb_floating_single.run(twotone)
#    pfb_fixed_single.run(fsig1)
#    outputpowfloat[st-1] = np.sum(np.abs(pfb_floating_single.X_k[:,-1])**2)
#    outputpowfix[st-1] = np.sum(np.abs(pfb_fixed_single.X_k.to_complex()[:,-1])**2)
#    st+=1
#
#fig = plt.figure()
#plt.plot(inputpow, outputpowfloat,'k')
#plt.savefig('../snapsf_engine/PFB_twotone/shiftzerod/twotone_power_flt_10.png')
#plt.title('Floating point power spectrum: shiftzerod')
#plt.xlabel('input power - $\Sigma | input |^{2}$')
#plt.ylabel('output power - $\Sigma | output |^{2}$')
#plt.show()
#fig = plt.figure()
#plt.plot(inputpow, outputpowfix,'k')
#plt.title('Fixed point power spectrum: shiftzerod')
#plt.xlabel('input power - $\Sigma | input |^{2}$')
#plt.ylabel('output power - $\Sigma | output |^{2}$')
#plt.savefig('../snapsf_engine/PFB_twotone/shiftzerod/twotone_power_fxd_10.png')
#plt.show()



#error = np.abs(outputpowfloat-outputpowfix)
    
#pfb_fixed_single.run(fsig1)
#plt.plot(np.abs(pfb_fixed_single.X_k.to_complex()))
#plt.show()
    
#fig, ax = plt.plot(2, 1, sharex=False, sharey=False)
#ax[0].hist(np.abs(pfb_floating_single.X_k[:,-1]),255,color='k')
#ax[0].set_title('floating-point, stage: '+str(st))
#ax[1].hist(np.abs(pfb_fixed_single.X_k.to_complex()[:,-1]),255,color='k')
#ax[1].set_title('fixed-point, stage: '+str(st))
#fig.savefig('../snapsf_engine/PFB_twotone/shiftcanada/unpowered/hist/st'+str(st)+'_twotone_hist.png')
#fig.clear()
