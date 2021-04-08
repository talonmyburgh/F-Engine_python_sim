using FFTW;
using Test;
using Plots;
N=2048;
include("pfb_floating.jl");

twiddle = makeTwiddle(N);
revTwid = bitRevArray(twiddle, size(twiddle)[1]);

data = zeros(ComplexF64,N);
data[3] = 1.0 + 1.0*im;
ourfft = natInIterDitFFT(data,revTwid);  
plot(collect(1:N), real.(ourfft))
idealfft = fft(data);
plot(collect(1:N), real.(idealfft))
@test all(abs.(idealfft .- ourfft).<0.0001);