# Binary Output
using DSP
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

function nextpow2(val::Float64) :: Integer
    i = 0;
    while true
        if 2^i >= val
            return i;
        else
            i+=1;
        end
    end
end