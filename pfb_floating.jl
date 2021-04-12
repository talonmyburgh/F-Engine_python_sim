include("pfb_coeff_gen.jl");
# =============================================================================
# Bit reversal algorithms used for the iterative fft's data re-ordering
# =============================================================================
# Arrange values from chronological to bit-reversed 
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
end

# Takes an array of length N which must be a power of two
function bitRevArray(array::Array{<:Complex{<:Real}}, N::Integer)::Array{<:Complex{<:Real}}
    return array[bitRev(N)];
end

# =============================================================================
# FFT: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================
# Generate Twiddle factors
function makeTwiddle(N::Integer)::Array{<:Complex{<:Real}}
    i = collect(0:div(N, 2)-1);
    twids = exp.(i .* (-2 * pi * (1im / N)));
    return twids;
end

function natInIterDitFFT(data::Array{<:Complex{<:Real}}, twids::Array{<:Complex{<:Real}}; staged::Bool=false)::Array{<:Complex{<:Real}}
    data = copy(data);
    N = size(data)[1];
    if staged
        stdg_data = zeros(ComplexF64, N, convert(Int64, log2(N)) + 2);
        stdg_data[:,1] .= data;
    end
    num_of_groups = 1;
    distance = div(N,2);
    stg = 1;
    while num_of_groups < N
        for k in 0:num_of_groups-1
            jfirst = (2 * k * distance) + 1;
            jlast = jfirst + distance - 1;
            W = twids[k+1];
            slc1 = jfirst:jlast;
            slc2 = slc1 .+ distance;
            tmp = W .* data[slc2];
            data[slc2] .= data[slc1] .- tmp;
            data[slc1] .= data[slc1] .+ tmp;
        end
        num_of_groups *= 2;
        distance = div(distance, 2);
        if staged
            stgd_data[:,stg] = data[:];
        end
        stg += 1;
    end
    if staged
        stgd_data[:,end] .= bitRevArray(stdg_data[:,end - 1], N);
        return stgd_data;
    else
        return bitRevArray(data, N);
    end
end

# =============================================================================
# Floating point PFB implementation making use of the natural order in fft
# like CASPER does. 
# =============================================================================

struct FloatPFBScheme
    N           :: Integer;
    dual        :: Bool;
    reg         :: Array{ComplexF64};
    staged      :: Bool;
    fwidth      :: Float64;
    chan_acc    :: Bool;
    window      :: Array{Float64};
    twids       :: Array{ComplexF64};
    function FloatPFBScheme(N::Integer, taps::Integer, w::String, dual::Bool, 
                            staged::Bool, fwidth::Float64, chan_acc::Bool)
        new(N,
            dual,
            zeros(ComplexF64,N,taps),
            staged,fwidth,
            chan_acc,
            coeff_gen(N, taps, win=w, fwidth=fwidth)[1],
            bitRevArray(makeTwiddle(N),div(N,2))
        );
    end
end