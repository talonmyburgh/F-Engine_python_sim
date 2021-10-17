include("pfb_coeff_gen.jl");
include("Fixpoint.jl");

"""
Prepares a BitVector for the shiftregister (swreg). It can take in
a decimal (as an Integer), a hex or oct value, or  a Vector{Bool}. 
"""
function shiftregMaker(val :: Union{Integer, Vector{<:Integer}, Vector{Bool}}, N :: Integer) :: BitVector
    nof_stages = Int(log2(N));
    if typeof(val) <: Integer
        return BitVector(reverse(digits(val,base=2,pad=nof_stages)));
    else
        val_len = length(val);
        if val_len <= nof_stages
            return val_len == nof_stages ? BitVector(val) : BitVector(append!(zeros(nof_stages-val_len),val));
        else
            error("Boolean Vector provided has length greater than log2(N)");
        end
    end
end

"""
Bit reverses an array's contents according to bitRev of the array's indices.
Takes an array of length N which must be a power of two.
"""
function fixBitRevArray(array::CFixpoint, N::Integer)::CFixpoint
    return array[bitRev(N)];
end;

"""
Generates twiddle factors for the natInIterDitFFT FFT.
See also: [`fixNatInIterDitFFT`](@ref)   
"""
function fixMakeTwiddle(N::Integer, fx_scheme::FixpointScheme)::CFixpoint
    i = collect(0:div(N, 2)-1);
    twids = exp.(i .* (-2 * pi * (1im / N)));
    fx_twids = fromComplex(twids,fx_scheme);
    return fx_twids;
end

"""
Fixed point PFB implementation that makes use of the fixed point natural order in fft
like CASPER does.
"""

struct FixPFBScheme
    N           ::Integer;
    dual        ::Bool;
    reg         ::CFixpoint;
    staged      ::Bool;
    fwidth      ::Float64;
    chan_acc    ::Bool;
    window      ::Fixpoint;
    twids       ::CFixpoint;
    swreg       ::BitVector;
    in_dat_sch  ::FixpointScheme;
    win_dat_sch ::FixpointScheme;
    coef_dat_sch::FixpointScheme;
    stg_dat_sch ::FixpointScheme;
    out_dat_sch ::FixpointScheme;
    function FixPFBScheme(N::Integer, taps::Integer, in_dat_sch::FixpointScheme, 
                            win_dat_sch::FixpointScheme, coef_dat_sch::FixpointScheme, 
                            stg_dat_sch::FixpointScheme, out_dat_sch::FixpointScheme; 
                            w::String="hanning", dual::Bool=false, 
                            swreg::Union{Integer, Vector{Bool}, BitVector}=N-1, 
                            staged::Bool=false, fwidth::Float64=1.0, chan_acc::Bool=false)
        new(N,
            dual,
            zeros(in_dat_sch,(N,taps),complex=true),
            staged,
            fwidth,
            chan_acc,
            fromFloat(coeff_gen(N, taps, win=w, fwidth=fwidth)[1], win_dat_sch),
            fixBitRevArray(fixMakeTwiddle(N,coef_dat_sch),div(N,2)),
            shiftregMaker(swreg,N),
            in_dat_sch, 
            win_dat_sch,
            coef_dat_sch, 
            stg_dat_sch, 
            out_dat_sch
        );
    end
end

"""
``` 
fixNatInIterDitFFT(pfbsch::FloatPFBScheme, data::Array{<:Complex{<:Real}}) :: Array{<:Complex}
```
natInIterDitFFT accepts PFB scheme and data.
"""
function fixNatInIterDitFFT(fixpfbsch::FixPFBScheme, c_data::CFixpoint) :: CFixpoint
    c_data = quantise(c_data,fixpfbsch.stg_dat_sch); #make it stg_data size, not in_data size
    N = length(c_data.real.data);
    if fixpfbsch.staged
        stgd_data = zeros(fixpfbsch.stg_dat_sch, N, convert(Int64,log2(N))+2);
        stgd_data[:,1] .= c_data; 
    end
    num_of_groups = 1;
    distance = div(fixpfbsch.N,2);
    stg = 2;
    while num_of_groups < fixpfbsch.N
        for k  in 0:num_of_groups-1
            jfirst = (2* k * distance) + 1;
            jlast = jfirst + distance - 1;
            W = fixpfbsch.twids[k + 1];
            slc1 = jfirst:jlast;
            slc2 = slc1 .+ distance;
            tmp = W * c_data[slc2];
            tmp = tmp >> (fixpfbsch.coef_dat_sch.fraction);
            tmp = cast(tmp,fixpfbsch.stg_dat_sch);
            tmp = normalise(tmp);
            c_data[slc2] = c_data[slc1] - tmp;
            c_data[slc1] = c_data[slc1] + tmp;
        end
        if fixpfbsch.swreg[stg-1]
            c_data = c_data >> 1;
        end
        c_data = cast(c_data,fixpfbsch.stg_dat_sch);
        c_data = normalise(c_data);

        num_of_groups *= 2;
        distance = div(distance, 2);

        if fixpfbsch.staged
            stgd_data[:,stg] = c_data[:];
        end
        stg+=1;
    end
    if fixpfbsch.staged
        stgd_data[:,end] .= fixBitRevArray(stgd_data[:,end - 1], fixpfbsch.N);
        return stgd_data;
    else
        return fixBitRevArray(c_data, fixpfbsch.N);
    end
end


