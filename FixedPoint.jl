"""
Structure containing information about the fixedpoint data scheme.
This scheme accompanies an integer array and governs all arithmetic
operations performed on it.
"""
struct FixedPointScheme
    bits::Integer
    fraction::Integer
    min :: Integer
    max :: Integer
    unsigned :: Bool
    range :: Integer
    scale :: Integer
    overflow_clip :: Bool
    underflow_round :: RoundingMode
    function FixedPointScheme(bits :: Integer,
                                fraction :: Integer,
                                min_int::Union{Integer, Nothing}=nothing,
                                max_int::Union{Integer, Nothing}=nothing,
                                unsigned::Bool=false,
                                overflow_clip::Bool=false,
                                underflow_round::RoundingMode=RoundNearest)
        scale = 2^fraction
        range = 2^bits
        min = min_int===nothing ? unsigned ? 0 : (-range//2) : min_int;
        max = max_int===nothing ? unsigned ? range - 1 : (range//2) -1 : max_int
        return new(bits, fraction, min, max, unsigned, range, scale, overflow_clip, underflow_round)      
    end
end

"""
The FixedPoint structure contains a FixedPointScheme and Integer Array
for full Fixed Point number representation.
"""
struct FixedPoint{T , N} <: AbstractArray{T, N}
    data
    scheme :: FixedPointScheme
    function FixedPoint(data :: Array{T, N}, scheme ::FixedPointScheme) where {T,N}
        Z = T <: Real ? Integer : Complex{Integer} 
        return new{Z, N}(convert.(Z,
                scheme.overflow_clip ? 
                clamp.(round.(data, scheme.underflow_round), scheme.min, scheme.max)
                :
                clamp_wrap.(round.(data, scheme.underflow_round), scheme.min, scheme.max)
            ), scheme
        )
    end
    function FixedPoint(data :: T, scheme :: FixedPointScheme) where {T}
        Z = T <: Real ? Integer : Complex{Integer}
        return new{Z, 1}(convert.(Z, 
                scheme.overflow_clip ? 
                [clamp(round(data, scheme.underflow_round), scheme.min, scheme.max)]
                :
                [clamp_wrap(round(data, scheme.underflow_round), scheme.min, scheme.max)]
            ), scheme
        )
    end
end

"""
Returns size of inner data array
"""
Base.size(Fx :: FixedPoint) = size(Fx.data)

"""
Getindex functions for the FixedPoint type. Indexing style is cartesian, and returns 
the Integer data inside. This means a loss of the associated scheme so be sure to store it.
"""
Base.getindex(Fx :: FixedPoint, I) = Fx.data[I]
Base.getindex(Fx :: FixedPoint, I...) = Fx.data[I...]

"""
Setindex functions for the FixedPoint type. A check on the scheme is made to ensure they are the 
same before integer elements from one Fixed Point instance are copied to the other.
"""
function Base.setindex!(Fx :: FixedPoint, v :: FixedPoint, I :: UnitRange)
    if Fx.scheme != v.scheme
        error("Schemes of two Fixpoint elements are not equivalent. Copying data fields across anyway.")
    end
    return Fx.data[I] = v.data[1:end]
end
function Base.setindex!(Fx :: FixedPoint, v :: FixedPoint, I :: Integer) 
    if Fx.scheme != v.scheme
        error("Schemes of two Fixpoint elements are not equivalent. Copying data fields across anyway.")
    end
    return Fx.data[I] = v.data[1]
end
Base.setindex!(Fx :: FixedPoint, v, I) = (Fx.data[I] = v)

"""
Length function returns the length of the internal integer array.
"""
Base.length(Fx :: FixedPoint) = length(Fx.data)

"""
Similar function returns a FixedPoint type of equal dimensions.
"""
Base.similar(Fx :: FixedPoint, ::Type{T}, dims::Dims) where {T} = FixedPoint(similar(Fx.data, T, dims), Fx.sign)

"""
An overload of clamp_wrap to take a Fixpoint array argument instead of an Integer.
"""
function clamp_wrap(f :: Fixpoint, min :: Integer, max :: Integer)
    clamp_val = ((f.data .- min) .% (min - max)) .+ min;
    scheme = FixpointScheme(f.scheme.bits,f.scheme.fraction,unsigned=f.scheme.unsigned,
    max_int=max, min_int=min,
    ovflw_behav=f.scheme.ovflw_behav, undflw_behav=f.scheme.undflw_behav);
    return Fixpoint(clamp_val,scheme);        
end

"""
Does a clamp operation but wraps the value to min/max rather than saturate 
the value like standard clamp.
"""
function clamp_wrap(i :: Integer, min :: Integer, max :: Integer)
    return ((i - min) % (max - min)) + min;
end