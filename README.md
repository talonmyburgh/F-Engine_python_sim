# F-Engine_python_sim
For my Masters Degree, I have created an F-Engine simulator. 
The purpose of this simulator is to enable the quick testing and prototyping of various fixed-point and floating-point Polyphase Filterbanks.
These Polyphase Filterbanks are modeled around the CASPER (https://casper-toolflow.readthedocs.io/en/latest/) PFB implementation that is used in many of the worlds leading radio observatories. The motivation for this research was the debugging of MeerKAT's F-Engine. CASPER's PFB is comprised of a FIR filterbank proceeded by a Radix 2, natural order in FFT.
Aspects of the PFB are generic however in that you may adjust point size, tap size, windowing coefficients, data type processed, bit size etc. For more on usage see Jupyter notebook file "HOWTO.ipybn".
