# -*- coding: utf-8 -*-
"""
@author: talonmyburgh
"""
import numpy as np
from pfb_fixed import FixPFB
from pfb_floating import FloatPFB
from fixpoint import cfixpoint
import matplotlib.pyplot as plt

bits_out =17
bits_in = 9
method = "ROUND"
taps = 16
point = 2048 #2k
N=2**16
k=np.arange(N)

####SIGNALS######
#sig1 = np.cos(32*(2*np.pi/N)*k)
#sig2 = np.cos(32*(2*np.pi/N)*64*k)
#sig3 = np.cos(32*(2*np.pi/N)*128*k)
#sig4 = np.cos(32*(2*np.pi/N)*192*k)
#sig5 = np.cos(32*(2*np.pi/N)*64*4*k)
#sig6 = np.cos(32*(2*np.pi/N)*64*5*k)
#sig7 = np.cos(32*(2*np.pi/N)*64*6*k)
#sig8 = np.cos(32*(2*np.pi/N)*64*7*k)
#sig9 = np.cos(32*(2*np.pi/N)*64*8*k)
#sig10 = np.cos(32*(2*np.pi/N)*64*9*k)
#sig11 = np.cos(32*(2*np.pi/N)*64*10*k)
#sig12 = np.cos(32*(2*np.pi/N)*64*11*k)
#sig13= np.cos(32*(2*np.pi/N)*64*12*k)
#sig14 = np.cos(32*(2*np.pi/N)*64*13*k)
#sig15 = np.cos(32*(2*np.pi/N)*64*14*k)
#sig16 = np.cos(32*(2*np.pi/N)*64*15*k)

impulse = np.zeros(N)
slc =  slice(1,N,2048)
impulse[slc] = 1
#
#np.save('impulsesiminput.npy',impulse/2.2)
#np.save('impulsefpgainput.npy',impulse/1.1)

#twotonesiminput = (sig1+sig2)/4.4
#twotonefpgainput = (sig1+sig2)/2.2

#sixteentonesiminput = (sig1+sig2+sig3+sig4+sig5+sig6+sig7+sig8+sig9+sig10+sig11+sig12+sig13+sig14+sig15+sig16)/(16*2.2)
#sixteentonefpgainput = (sig1+sig2+sig3+sig4+sig5+sig6+sig7+sig8+sig9+sig10+sig11+sig12+sig13+sig14+sig15+sig16)/(16*1.1)


#plt.plot(sig5[:2048])
#plt.show()
#plt.plot(sig3[:2048])
#plt.show()
#np.save('sixteentonesiminput.npy',sixteentonesiminput)
#np.save('sixteentonefpgainput.npy',sixteentonefpgainput)

#fsig = cfixpoint(bits_in,bits_in,method = method)
#fsig2 = cfixpoint(bits,fraction,method = method)
#fsig3 = cfixpoint(bits,fraction,method = method)
#frnd = cfixpoint(bits,fraction,method = method)
#datar = np.load('impulsesiminput.npy')
#fsig.from_complex(datar)
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
#shiftreg = [1,1,1,1,1,1,1,1,1,1,1]
#shiftreg = [0,1,0,1,0,1,0,1,0,1]
#shiftreg = [0,0,0,0,0,1,1,1,1,1]
#shiftreg = [1,1,1,1,1,0,0,0,0,0]

#pfb_floating_single = FloatPFB(point,taps)
#pfb_floating_single.run(datar)
#np.save("F_Enginesim_float_impulse_out.npy",pfb_floating_single.X_k)

#pfb_fixed_single = FixPFB(point,taps,bits_in,bits_out,shiftreg)
#pfb_fixed_single.run(fsig)
#np.save("F_Enginesim_fixed_impulse_out.npy",pfb_fixed_single.X_k.to_complex())

#fpgadatazero = np.load('raw_data_oned.npy')
fpgadata = np.load('raw_data_impulse_out.npy')
#simdatazero = np.load('F_Enginesim_fixed_out_twotone_oned_FIRchange.npy')
#simdatazerochange = np.load('F_Enginesim_fixed_out_twotone_oned_FIRchange_roundchange.npy')
simdata = np.load('F_Enginesim_fixed_impulse_out.npy')
simfloat = np.load('F_Enginesim_float_impulse_out.npy')
#
plt.figure(1)
plt.subplot(311)
plt.title('FPGA vs Simulator Real data for sixteen tone input\n and zero\'d shift register')
plt.xlabel('bins')
plt.ylabel('$Re\{output\}$')
plt.plot(np.real(simdata)[:,-1][:1024], 'k',label = 'fxdpt sim f-engine data')
plt.legend()
plt.subplot(312)
plt.xlabel('bins')
plt.ylabel('$Re\{output\}$')
plt.plot(np.real(simfloat)[:,-1][:1024], 'k', label = 'fltpt sim f-engine data')
plt.legend()
plt.subplot(313)
plt.xlabel('bins')
plt.ylabel('$Re\{output\}$')
plt.plot(np.real(fpgadata), 'k', label = 'fpga f-engine data')
plt.legend()
plt.show()
plt.figure(2)
plt.subplot(311)
plt.title('FPGA vs Simulator Imag data for sixteen tone input\n and zero\'d shift register')
plt.xlabel('bins')
plt.ylabel('$Im\{output\}$')
plt.plot(np.imag(simdata)[:,-1][:1024], 'k',label = 'fxdpt sim f-engine data')
plt.legend()
plt.subplot(312)
plt.xlabel('bins')
plt.ylabel('$Im\{output\}$')
plt.plot(np.imag(simfloat)[:,-1][:1024], 'k', label = 'fltpt sim f-engine data')
plt.legend()
plt.subplot(313)
plt.xlabel('bins')
plt.ylabel('$Im\{output\}$')
plt.plot(np.imag(fpgadata), 'k', label = 'fpga f-engine data')
plt.legend()
plt.show()
plt.figure(3)
plt.subplot(311)
plt.title('FPGA vs Simulator Absolute data for sixteen tone input\n and zero\'d shift register')
plt.xlabel('bins')
plt.ylabel('$|output|$')
plt.plot(np.abs(simdata)[:,-1][:1024], 'k',label = 'fxdpt sim f-engine data')
plt.legend()
plt.subplot(312)
plt.xlabel('bins')
plt.ylabel('$|output|$')
plt.plot(np.abs(simfloat)[:,-1][:1024], 'k', label = 'fltpt sim f-engine data')
plt.legend()
plt.subplot(313)
plt.xlabel('bins')
plt.ylabel('$|output|$')
plt.plot(np.abs(fpgadata), 'k', label = 'fpga f-engine data')
plt.legend()
plt.show()


#plt.figure(2)
#plt.subplot(211)
#plt.title('FPGA vs Simulator Absolute of the data \n for oned shift register')
#plt.xlabel('bins')
#plt.ylabel('$|output|$')
#plt.plot(np.abs(simdataoned)[:1024], 'k',label = 'simulator f-engine data')
#plt.legend()
#plt.subplot(212)
#plt.xlabel('bins')
#plt.ylabel('$|output|$')
#plt.plot(np.abs(fpgadataoned), 'k', label = 'fpga f-engine data')
#plt.legend()
#plt.show()

#plt.figure(3)
#plt.subplot(211)
#plt.title('Floating point PFB real and imag results for the two tone\n signal processing')
#plt.xlabel('bins')
#plt.ylabel('$Re\{output\}$')
#plt.plot(np.real(simfloat)[:1024], 'k')
#plt.legend()
#plt.subplot(212)
#plt.xlabel('bins')
#plt.ylabel('$Im\{output\}$')
#plt.plot(np.imag(simfloat)[:1024], 'k')
#plt.legend()
#plt.show()


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
#plt.plot(np.abs(pfb_fixed_single.X_k.to_complex()[:,-1][:1024]),'k')
#plt.title('Fixed point spectrum')
#plt.xlabel('n')
#plt.ylabel('$|output|^{2}$')
##plt.savefig('../snapsf_engine/PFB_twotone/shiftzerod/twotone_power_fxd_10.png')
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
