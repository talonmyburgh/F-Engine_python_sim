# -*- coding: utf-8 -*-
import numpy as np
import casperfpga
import pickle

##########FPGA setup###########
fpga =  casperfpga.CasperFpga('192.168.14.94')
fpga.upload_to_ram_and_program('pfb_test.fpg')
fpga.registers.fft_shift.write(reg=0)

#######Sort Data placement#####
data = np.load('impulsefpgainput.npy')
outputdict={"read_init":{"valid_count":0,"of_count":0,"real":np.zeros(4,dtype=np.float64),"imag":np.zeros(4,dtype=np.float64)}}

##########Initialise###########
fpga.registers.control.write(en=1)
fpga.registers.control.write(sync=1)

##########Loops################
fpga.registers.control.write(sync=0)
i=0
processing=True
while processing:
    fpga.registers.control.write(en=0)
    v= i*8
    valid=fpga.registers.status.read()['data']['valid_count']
    print("Valid=",valid)
    if(valid!=0):
        fpga.registers.din0.write(reg = data[v])
        fpga.registers.din1.write(reg = data[v+1])
        fpga.registers.din2.write(reg = data[v+2])
        fpga.registers.din3.write(reg = data[v+3])
        fpga.registers.din4.write(reg = data[v+4])
        fpga.registers.din5.write(reg = data[v+5])
        fpga.registers.din6.write(reg = data[v+6])
        fpga.registers.din7.write(reg = data[v+7])
        fpga.registers.control.write(en=1)          #process data.

    else:
        processing=False
        fpga.registers.control.write(en=1)
        break
    i+=1

capturing=True

j=0

while capturing:
    print(j)
    fpga.registers.control.write(en=0)
    reals = np.array([fpga.registers.real0.read()['data']['reg'],
                      fpga.registers.real1.read()['data']['reg'],
                      fpga.registers.real2.read()['data']['reg'],
                      fpga.registers.real3.read()['data']['reg']],dtype = np.float64)
    imags = np.array([fpga.registers.imag0.read()['data']['reg'],
                      fpga.registers.imag1.read()['data']['reg'],
                      fpga.registers.imag2.read()['data']['reg'],
                      fpga.registers.imag3.read()['data']['reg']],dtype = np.float64)
    outputdict["read_"+str(j)] = {
            "valid_count":fpga.registers.status.read()['data']['valid_count'],
            "of_count":fpga.registers.status.read()['data']['of_count'],
            "real":reals,"imag":imags}
    if(j==255):
        capturing=False
        break
    j+=1
    fpga.registers.control.write(en=1)

#########Save Dictionary#########

pickle_out = open("F_Engine_out_impulse.pickle","wb")
pickle.dump(outputdict, pickle_out)
pickle_out.close()

#########Save raw data npy file####

raw_data = np.zeros(1024,dtype = np.complex)
v=0
for i in range(len(outputdict)-1):
    print(i,len(outputdict))
    slc = slice(v,v+4)
    tmpdict = outputdict["read_"+str(i)]
    treal = tmpdict['real']
    timag = tmpdict['imag']
    raw_data[slc] = treal[:]+1j*timag[:]
    v+=4
np.save('raw_data_impulse_out',raw_data)


