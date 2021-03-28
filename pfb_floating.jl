include("pfb_coeff_gen.jl");
# =============================================================================
# Bit reversal algorithms used for the iterative fft's data re-ordering
# =============================================================================
#Arrange values from chronological to bit-reversed 
function bitRev(val::Array{Int}, bits::Integer) :: Array{Int}
    val_copy = copy(val);
    N = 1<<bits;
    for i=1:bits-1
        val .>>= 1;
        val_copy .<<= 1;
        val_copy .|= (val.&1);
    end
    val_copy = val_copy .& (N-1);
    return val_copy;
end

#Takes an array of length N which must be a power of two
function bitRevArray(array::Array{Complex{Float64}}, N::Integer) :: Array{Complex{Float64}}
    bits = Int(log2(N));
    A = Array{Complex{Float64}}(undef,N);
    a = collect(0:N-1);
    rev_a = bitRev(a,bits);
    for i=1:N
        A[rev_a[i]+1] = array[i];
    end
    return A;
end