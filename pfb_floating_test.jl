using FFTW;
using Test;
using PyPlot;
pygui(true);
include("pfb_floating.jl");

# Test bit reversal
s = collect(0:16-1) .+ im.*collect(0:16-1);
t = bitRevArray(s,16);
t = bitRevArray(t,16);
@test s == t;

# Test FFTs
N=32;
k = collect(1:1:N);
taps=4;
pfbsch = FloatPFBScheme(N,taps);
data = cos.(6*(2*pi/N) .*k);
data = data .+ 1im .* zeros(N,1);
ourfft = natInIterDitFFT(pfbsch, data);  
idealfft = fft(data);
@test all(abs.(idealfft .- ourfft).<0.0001);
 
# Test Separation
data = zeros(ComplexF64,N);
data[15] = 1.0 + 0.0*im;
idealfft = fft(data);
(G_k,H_k) = SpecSplit(idealfft);
g_k = fft(real(data));
h_k = fft(imag(data));
@test all(abs.(g_k .- G_k).<1e-6);
@test all(abs.(h_k .- H_k).<1e-6);

# Test Power
# single fft output
pow_ourfft = SpecPow(pfbsch, ourfft);
ideal_pow_fft = (abs.(ourfft)).^2;
@test all((pow_ourfft .- ideal_pow_fft).<1e-6);

# #Test PFB
n=N*taps*10;
k=collect(0:n-1);
sig1 = cos.(32*(2*pi/n) .*k);
sig2 = cos.(32*(2*pi/n)*64 .*k);
sig3 = cos.(32*(2*pi/n)*128 .*k);
sig4 = cos.(32*(2*pi/n)*192 .*k);
sig5 = cos.(32*(2*pi/n)*64*4 .*k);
sig6 = cos.(32*(2*pi/n)*64*5 .*k);
sig7 = cos.(32*(2*pi/n)*64*6 .*k);
sig8 = cos.(32*(2*pi/n)*64*7 .*k);
sig9 = cos.(32*(2*pi/n)*64*8 .*k);
sig10 = cos.(32*(2*pi/n)*64*9 .*k);
sig11 = cos.(32*(2*pi/n)*64*10 .*k);
sig12 = cos.(32*(2*pi/n)*64*11 .*k);
sig13= cos.(32*(2*pi/n)*64*12 .*k);
sig14 = cos.(32*(2*pi/n)*64*13 .*k);
sig15 = cos.(32*(2*pi/n)*64*14 .*k);
sig16 = cos.(32*(2*pi/n)*64*15 .*k);
sixteentonesiminput = Complex.((sig1.+sig2.+sig3.+sig4.+sig5.+sig6.+sig7.+sig8.+sig9.+sig10.+sig11.+sig12.+sig13.+sig14.+sig15.+sig16)./(16*2.2));

#For starters:
pfbsch = FloatPFBScheme(N,taps,w="hanning",dual=false,staged=false,chan_acc=false);
X_k = RunPFB(pfbsch,data=sixteentonesiminput);
plot(X_k[:,end])
#For seconds:
pfbsch = FloatPFBScheme(N,taps,w="hanning",dual=false,staged=false,chan_acc=true);
X_k = RunPFB(pfbsch,data=sixteentonesiminput);
plot(X_k)
#For thirds:
pfbsch = FloatPFBScheme(N,taps,w="hanning",dual=true,staged=false,chan_acc=false);
(G_k,H_k) = RunPFB(pfbsch,data=sixteentonesiminput);
plot(abs.(H_k[:,1]))
#For fourths:
pfbsch = FloatPFBScheme(N,taps,w="hanning",dual=true,staged=true,chan_acc=false);
(G_k,H_k) = RunPFB(pfbsch,data=sixteentonesiminput);
plot(H_k[:,end])