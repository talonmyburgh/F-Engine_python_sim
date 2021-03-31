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
    return a_copy;
end

# Takes an array of length N which must be a power of two
function bitRevArray(array::Array{<:Complex{<:Real}}, N::Integer)::Array{<:Complex{<:Real}}
    A = Array{Complex{Real}}(undef, N);
    rev_a = bitRev(N);
    A[rev_a .+ 1] .= array;
    return A;
end

# =============================================================================
# FFT: natural data order in, bit reversed twiddle factors, bit reversed 
# order out.
# =============================================================================
# Generate Twiddle factors
function makeTwiddle(N::Integer)::Array{<:Complex{<:Real}}
    i = collect(0:div(N, 2));
    twids = exp.(i .* (-2 * pi * (1im / N)));
    return twids;
end

function natInIterDitFFT(data::Array{<:Complex{<:Real}}, twids::Array{<:Complex{<:Real}}; staged::Bool=false)::Array{<:Complex{<:Real}}
    N = size(data)[1];
    if staged
        stdg_data = zeros(Complex64, N, convert(Int64, log2(N)) + 2);
        stdg_data[:,1] = data[:];
    end
    num_of_groups = 1;
    distance = N // 2;
    stg = 1;
    while num_of_groups < N
        for k = 0:num_of_groups - 1
            jfirst = 2 * k * distance;
            jlast = jfirst + distance - 1;
            W = twids[k + 1];
            slc1 = jfirst:jlast;
            slc2 = jfirst + distance:jlast + distance;
            tmp = W .* data[slc2];
            data[slc2] .= data[slc1] .- tmp;
            data[slc1] .= data[scl1] .+ tmp;
        end
        num_of_groups *= 2;
        distance = div(distance, 2);
        if staged
            stgd_data[:,stg] = data[:];
        end
        stg += 1;
        if staged
            stgd_data[:,end - 1] .= bitRevArray(stdg_data[:,end - 2], N);
            return stgd_data;
        else
            return bitRevArray(data, N);
        end
    end