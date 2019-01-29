# -*- coding: utf-8 -*-
import numpy as np
import casperfpga
import pickle
import sys

##########FPGA setup###########
board = sys.argv[1]                                 #specify IP of board to program
image = sys.argv[2]                                 #specify image to program FPGA
inputdata = sys.argv[3]                             #specify data to process
stackcount = int(sys.argv[4])*128                        #specify how many full reads are required
floatorint = bool(sys.argv[5])                            #process float or integer
fpga =  casperfpga.CasperFpga(board)                #configure board as object
fpga.upload_to_ram_and_program(image)               #upload image to board
fpga.registers.fft_shift.write(reg=0)               #populate FFT shift registers
if(floatorint):
    fpga.registers.control.write(src_mux=0)             #decide on whether to pass floats (0) or ints (1)
    DTYPE = np.float
else:
    fpga.registers.control.write(src_mux=1)             #decide on whether to pass floats (0) or ints (1)
    DTYPE=np.int32

#######Sort Data placement#####
data = np.load(inputdata)                           #fetch data from npy file
#Per read, we get 4 real and 4 imag results, which are stored in a dictionary with key = #read
outputdict={"read_init":{"valid_count":0,"of_count":0,"real":np.zeros(4,dtype=DTYPE),"imag":np.zeros(4,dtype=DTYPE)}}

##########Initialise###########
fpga.registers.control.write(en=1)                  #set first enable
fpga.registers.control.write(sync=1)                #push in sync, so we have indicator for when first read is ready

##########Loops################
fpga.registers.control.write(sync=0)
inputdataindex=0
writing = True
while writing:
        fpga.registers.control.write(en=0)
        v= inputdataindex*8
        valid=fpga.registers.status.read()['data']['valid_count']
        print("Valid=",valid)
        fpga.registers.din0.write(reg = data[v])
        fpga.registers.din1.write(reg = data[v+1])
        fpga.registers.din2.write(reg = data[v+2])
        fpga.registers.din3.write(reg = data[v+3])
        fpga.registers.din4.write(reg = data[v+4])
        fpga.registers.din5.write(reg = data[v+5])
        fpga.registers.din6.write(reg = data[v+6])
        fpga.registers.din7.write(reg = data[v+7])
        fpga.registers.control.write(en=1)          #load data.
        if(valid==0):
            writing = False
            writeandread = True
            break
        inputdataindex+=1
k=0
stackcnt = 0
while writeandread:
    print(k)
    fpga.registers.control.write(en=0)
    v= inputdataindex*8
    #read real and imag data as 4 reads each per time
    reals = np.array([fpga.registers.real0.read()['data']['reg'],
                      fpga.registers.real1.read()['data']['reg'],
                      fpga.registers.real2.read()['data']['reg'],
                      fpga.registers.real3.read()['data']['reg']],dtype = DTYPE)
    imags = np.array([fpga.registers.imag0.read()['data']['reg'],
                      fpga.registers.imag1.read()['data']['reg'],
                      fpga.registers.imag2.read()['data']['reg'],
                      fpga.registers.imag3.read()['data']['reg']],dtype = DTYPE)
    #store in a dictionary with key
    outputdict["read_"+str(k)] = {
            "valid_count":fpga.registers.status.read()['data']['valid_count'],
            "of_count":fpga.registers.status.read()['data']['of_count'],
            "real":reals,"imag":imags}
    #continue to write data to PFB
    fpga.registers.din0.write(reg = data[v])
    fpga.registers.din1.write(reg = data[v+1])
    fpga.registers.din2.write(reg = data[v+2])
    fpga.registers.din3.write(reg = data[v+3])
    fpga.registers.din4.write(reg = data[v+4])
    fpga.registers.din5.write(reg = data[v+5])
    fpga.registers.din6.write(reg = data[v+6])
    fpga.registers.din7.write(reg = data[v+7])
    fpga.registers.control.write(en=1)          #load data.
    
    if(stackcnt == stackcount):
        writeandread=False
        break
    
    stackcnt+=1
    inputdataindex +=1
    k+=1
    
#########Save Dictionary#########

pickle_out = open("F_Engine_output_"+inputdata[:-4]+".pickle","wb")
pickle.dump(outputdict, pickle_out)
pickle_out.close()

##Extract from Dictionary and save raw data npy file####

raw_data_real = np.zeros((1024,stackcount/128),dtype = DTYPE)
raw_data_imag = np.zeros((1024,stackcount/128),dtype = DTYPE)
v=0
stk=0
for i in range(len(outputdict)-1):
    print(i,len(outputdict))
    slc = slice(v,v+4)
    tmpdict = outputdict["read_"+str(i)]
    treal = tmpdict['real']
    timag = tmpdict['imag']
    raw_data_real[slc,stk] = treal[:]
    raw_data_imag[slc,stk] = timag[:]
    v+=4
    if(i%256==0):
        stk+=1
        v=0
np.save('raw_data_'+inputdata[:-4]+'_real',raw_data_real)
np.save('raw_data_'+inputdata[:-4]+'_imag',raw_data_imag)
    
        
        
        
        
        
        