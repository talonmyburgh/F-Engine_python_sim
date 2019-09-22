#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:22:17 2019

@author: talon
"""

import numpy as np
import casperfpga
import pickle

##########FPGA setup###########
fpga =  casperfpga.CasperFpga('skarab020A44-01')
fpga.upload_to_ram_and_program('meerkat_4k_pfb.fpg')
fpga.registers.fft_shift.write(reg=0)
fpga.registers.control.write(src_mux=0)

#######Sort Data placement#####
outputdict={"read_init":{"valid_count":0,"of_count":0,"real":np.zeros(4,dtype=np.float64),"imag":np.zeros(4,dtype=np.float64)}}


##########Initialise###########
fpga.registers.control.write(en=1)
fpga.registers.control.write(sync=1)

fpga.registers.control.write(sync=0)


i=0
j=0
processing=True
cnt = False
while processing:
    if(i==0):                                   #write a val to point 6 on 4k - should give sinusoid out.
        fpga.registers.din6.write(reg = 0.5)
    if(i==1):
        fpga.registers.din6.write(reg = 0.0)
    fpga.registers.control.write(en=0)
    valid=fpga.registers.status.read()['data']['valid_count']
    print("Valid=",valid,", iter =",i)
    fpga.registers.control.write(en=1)
    
    if(valid==0):
        cnt = True
    if(cnt):
        j+=1
    if(j==5):
        processing=False
        fpga.registers.control.write(en=1)
        break
    if(i==1024):
        i=0
    i+=1
    
capturing=True
k=0
while capturing:
    print(k)
    fpga.registers.control.write(en=0)
    reals = np.array([fpga.registers.real0.read()['data']['reg'],
                      fpga.registers.real1.read()['data']['reg'],
                      fpga.registers.real2.read()['data']['reg'],
                      fpga.registers.real3.read()['data']['reg']],dtype = np.float64)
    imags = np.array([fpga.registers.imag0.read()['data']['reg'],
                      fpga.registers.imag1.read()['data']['reg'],
                      fpga.registers.imag2.read()['data']['reg'],
                      fpga.registers.imag3.read()['data']['reg']],dtype = np.float64)
    outputdict["read_"+str(k)] = {
            "valid_count":fpga.registers.status.read()['data']['valid_count'],
            "of_count":fpga.registers.status.read()['data']['of_count'],
            "real":reals,"imag":imags}
    if(k==8191):
        capturing=False
        break
    fpga.registers.control.write(en=1)
    k+=1
    
#########Save Dictionary#########

pickle_out = open("F_4k_Engine_out_impulse.pickle","wb")
pickle.dump(outputdict, pickle_out)
pickle_out.close()

#########Save raw data npy file####

raw_data_real = np.zeros((32768),dtype = np.float64)
raw_data_imag = np.zeros((32768),dtype = np.float64)
v=0
for i in range(len(outputdict)-1):
    print(v,len(outputdict))
    slc = slice(v,v+4)
    tmpdict = outputdict["read_"+str(i)]
    raw_data_real[slc] = tmpdict['real'][:]
    raw_data_imag[slc] = tmpdict['imag'][:]
    v+=4

np.save('raw_4k_data_impulse_real',raw_data_real)
np.save('raw_4k_data_impulse_imag',raw_data_imag)
