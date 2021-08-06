include("pfb_coeff_gen.jl");
include("Fixpoint.jl");
"""
```
bitRevArray(array::Array{<:Complex{<:Real}}, N::Integer)::Array{<:Complex{<:Real}}
```
Bit reverses an array's contents according to bitRev of the array's indices.
Takes an array of length N which must be a power of two.
"""
function bitRevFixArray(array::CFixpoint, N::Integer)::CFixpoint
    return array[bitRev(N)];
end;