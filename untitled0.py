#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:38:13 2019

@author: talon
"""
import numpy as np
from pfb_fixed import FixPFB, iterffft_natural_DIT,make_fix_twiddle, bitrevfixarray
from pfb_floating import FloatPFB, iterfft_natural_in_DIT, make_twiddle, bitrevarray
import scipy.io as spio
import matplotlib.pyplot as plt
from fixpoint import cfixpoint,fixpoint

"""FFT plot comparisons"""
#fftinputdata = spio.loadmat('fftinputdata_ns.mat', squeeze_me=True)['inputdata']
#fftrealoutputdata = spio.loadmat('fftrealdataout_ns.mat', squeeze_me=True)['realoutputdata']
#fftimagoutputdata = spio.loadmat('fftimagdataout_ns.mat', squeeze_me=True)['imagoutputdata']
#b = make_twiddle(2**13)
#b = bitrevarray(b,2**12)
#a=make_fix_twiddle(2**13,17,16)
#a = bitrevfixarray(a,2**12)
#data = cfixpoint(17,17)
#data.from_complex(fftinputdata)
#fltfftres = np.real(iterfft_natural_in_DIT(fftinputdata,b))[:4096]
#casfftres = np.real(fftrealoutputdata+1j*fftimagoutputdata)
#fxdfftres = np.real(iterffft_natural_DIT(data,a,[1,1,1,1,1,1,1,1,1,1,1,1,1],17,17,17).to_complex())[:4096]
#
#plt.plot(fltfftres)
#plt.show()
#
#plt.plot(fxdfftres)
#plt.show()
#
#plt.plot(casfftres)
#plt.show()
#
#fxdmax = np.max(fxdfftres)
#casmax = np.max(casfftres)
#fltmax = np.max(fltfftres)
#print('fxdloc ',np.where(fxdfftres>fxdmax-0.001),' casloc ',np.where(casfftres>casmax-0.001), ' fltloc ',
#      np.where(fltfftres>fltmax-0.001))



"""FIR plot comparisons"""
firinputdata = spio.loadmat('firinputdata_ns.mat', squeeze_me=True)['inputdata']
firoutputdata = spio.loadmat('firoutputdata_ns.mat', squeeze_me=True)['outputdata']
pfbflt = FloatPFB(2**13,8)
pfbflt.run(firinputdata)
datfirfxd = cfixpoint(17,17)
datfirfxd.from_complex(firinputdata)
pfbfxd = FixPFB(2**13,8,17,17,17,17,8191,17,chan_acc=False,w='hann')
pfbfxd.run(datfirfxd)
firoutflt =np.sum(pfbflt.reg*pfbflt.window,axis=1)/ (2**pfbflt.firsc)
plt.plot(np.abs(firoutflt))
plt.show()

X_real = pfbfxd.reg_real*pfbfxd.window
X_imag = pfbfxd.reg_imag*pfbfxd.window
prodgrth = X_real.bits - pfbfxd.bits_fft-1
X = cfixpoint(real = X_real.sum(axis=1),imag = X_imag.sum(axis =1))                                 
X >> prodgrth + pfbfxd.firsc
X.bits = pfbfxd.bits_fft
X.fraction = pfbfxd.bits_fft
X.normalise()
plt.plot(np.abs(X.to_complex()))
plt.show()
plt.plot(np.abs(firoutputdata))
plt.show()

"""PFB plot comparisons"""
#pfbinputdata = spio.loadmat('pfbinputdata_ns.mat', squeeze_me=True)['inputdata']
#pfboutputrealdata = spio.loadmat('pfbrealdataout_ns.mat', squeeze_me=True)['realoutputdata']
#pfboutputimagdata = spio.loadmat('pfbimagdataout_ns.mat', squeeze_me=True)['imagoutputdata']
#pfbflt = FloatPFB(2**13,8)
#pfbflt.run(pfbinputdata)
#datpfbfxd = cfixpoint(17,17)
#datpfbfxd.from_complex(pfbinputdata)
#pfbfxd = FixPFB(2**13,8,17,17,17,17,8191,17,chan_acc=False,w='hann')
#pfbfxd.run(datpfbfxd)
#
#plt.plot(np.abs(pfbflt.X_k)[:,-1][:4096],'k', linewidth = 0.5)
#plt.savefig('floatpfb.png',dpi = 1000)
#plt.show()
#plt.plot(np.abs(pfbfxd.X_k.to_complex()[:,-1][:4096]),'k', linewidth = 0.5)
#plt.savefig('fxdpfb.png',dpi = 1000)
#plt.show()
#plt.plot(np.abs(pfboutputrealdata+1j*pfboutputimagdata),'k', linewidth = 0.5)
#plt.savefig('realpfb.png',dpi = 1000)
#plt.show()
#
#fxdmax = np.max(np.abs(pfbfxd.X_k.to_complex()[:,-1][:4096]))
#casmax = np.max(np.abs(pfboutputrealdata+1j*pfboutputimagdata))
#fltmax = np.max(np.abs(pfbflt.X_k)[:,-1][:4096])
#print('fxdloc ',np.where(np.abs(pfbfxd.X_k.to_complex()[:,-1][:4096])>fxdmax-0.001)
#,' casloc ',np.where(np.abs(pfboutputrealdata+1j*pfboutputimagdata)>casmax-0.001))
