using FFTW;
using Test;
using Plots;
#Test FFTs
N=2048;
include("pfb_floating.jl");
twiddle = makeTwiddle(N);
revTwid = bitRevArray(twiddle, size(twiddle)[1]);
data = zeros(ComplexF64,N);
data[3] = 1.0 + 1.0*im;
ourfft = natInIterDitFFT(data,revTwid);  
idealfft = fft(data);
@test all(abs.(idealfft .- ourfft).<0.0001);

#Test FIR
n=2048;
taps=4;
N=n*taps;
impulse = zeros(N);
slc = 81:n:N;
impulse[slc] .= 0.45;
impulse = Complex.(impulse);
pfbsch=FloatPFBScheme(n,taps);
r = PFBFir(pfbsch,impulse[1:n]);
r = PFBFir(pfbsch,impulse[n+1:2*n]);
r = PFBFir(pfbsch,impulse[2*n+1:3*n]);
r = PFBFir(pfbsch,impulse[3*n+1:4*n]);
plot(r)

#Test PFB
N=n*taps*10;
k=collect(0:N-1);
sig1 = cos.(32*(2*pi/N) .*k)
sig2 = cos.(32*(2*pi/N)*64 .*k)
sig3 = cos.(32*(2*pi/N)*128 .*k)
sig4 = cos.(32*(2*pi/N)*192 .*k)
sig5 = cos.(32*(2*pi/N)*64*4 .*k)
sig6 = cos.(32*(2*pi/N)*64*5 .*k)
sig7 = cos.(32*(2*pi/N)*64*6 .*k)
sig8 = cos.(32*(2*pi/N)*64*7 .*k)
sig9 = cos.(32*(2*pi/N)*64*8 .*k)
sig10 = cos.(32*(2*pi/N)*64*9 .*k)
sig11 = cos.(32*(2*pi/N)*64*10 .*k)
sig12 = cos.(32*(2*pi/N)*64*11 .*k)
sig13= cos.(32*(2*pi/N)*64*12 .*k)
sig14 = cos.(32*(2*pi/N)*64*13 .*k)
sig15 = cos.(32*(2*pi/N)*64*14 .*k)
sig16 = cos.(32*(2*pi/N)*64*15 .*k)
sixteentonesiminput = Complex.((sig1.+sig2.+sig3.+sig4.+sig5.+sig6.+sig7.+sig8.+sig9.+sig10.+sig11.+sig12.+sig13.+sig14.+sig15.+sig16)./(16*2.2))
#For starters:
pfbsch = FloatPFBScheme(n,taps,w="hanning",dual=false,staged=false,chan_acc=false);
X_k = RunPFB(pfbsch,data=sixteentonesiminput);
#For seconds:
pfbsch = FloatPFBScheme(n,taps,w="hanning",dual=false,staged=false,chan_acc=true);
X_k = RunPFB(pfbsch,data=sixteentonesiminput);
#For thirds:
pfbsch = FloatPFBScheme(n,taps,w="hanning",dual=true,staged=false,chan_acc=false);
(G_k,H_k) = RunPFB(pfbsch,data=sixteentonesiminput);
#For fourths:
pfbsch = FloatPFBScheme(n,taps,w="hanning",dual=true,staged=true,chan_acc=false);
(G_k,H_k) = RunPFB(pfbsch,data=sixteentonesiminput);