using DSP

"""
```
function coeff_gen(N::Integer, taps::Integer; win::String = "hanning", fwidth::Float64 = 1.0) ::Tuple{Array{Float64},Integer}
```
Generates the FIR coefficients for the prefilter. 
"""
function coeff_gen(N::Integer, taps::Integer; win::String = "hanning", fwidth::Float64 = 1.0) ::Tuple{Array{Float64},Integer}
    WinDic = Dict{String,Function}(                                                                     #dictionary of various filter types
    "hanning" => DSP.hanning,
    "hamming" => DSP.hamming,
    "bartlett" => DSP.bartlett,
    "blackman" => DSP.blackman,
    );
    alltaps = N*taps;
    windowval=WinDic[win](alltaps);                                               
    totalcoeffs = reshape(windowval.*sinc.(fwidth.*(collect(0:alltaps-1)./(N) .- taps/2)),N,taps);
    scalefac = nextpow2(maximum(sum(abs.(totalcoeffs),dims=2)));
    return totalcoeffs, scalefac;
end
"""
```
nextpow2(val::Float64)::Integer
```
Function that, given a certain integer, produces the next power of 2. 
"""
function nextpow2(val::Float64) :: Integer
    i = 0;
    while true
        if 2^i >= val
            return i;
        else
            i+=1;
        end
    end
end;

"""
```
bitRev(N::Integer)::Array{Int}
```
Bit reversal algorithms used for the iterative fft's data re-ordering.
Arranges values from chronological to bit-reversed. 
"""
function bitRev(N::Integer)::Array{Int}
    bits = Int(log2(N));
    a = collect(0:N - 1);
    a_copy = copy(a);
    N = 1 << bits;
    for i = 1:bits - 1
        a .>>= 1;
        a_copy .<<= 1;
        a_copy .|= (a .& 1);
    end
    a_copy = a_copy .& (N - 1);
    return a_copy .+ 1;
end;