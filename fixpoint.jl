import Base: sum, +, -, *, typemin, typemax

"""
```
struct FixpointScheme
    bits :: Integer
    fraction :: Integer
    min :: Integer
    max :: Integer
    unsigned :: Bool
    range :: Integer
    scale :: Integer
    ovflw_behav :: String
    undflw_behav :: String
```
Type which holds information about the Fixpoint value and influences its treatment
under arithmetic, logical and conversion operations.

See also: [`Fixpoint`](@ref)
"""
struct FixpointScheme 
    bits :: Integer
    fraction :: Integer
    min :: Integer
    max :: Integer
    unsigned :: Bool
    range :: Integer
    scale :: Integer
    ovflw_behav :: String
    undflw_behav :: String
    function FixpointScheme(bits::Integer, fraction::Integer;
         min_int::Union{Integer, Nothing}=nothing,max_int::Union{Integer, Nothing}=nothing,
         unsigned::Union{Bool, Nothing}=false, ovflw_behav::Union{String, Nothing}="WRAP",
         undflw_behav::Union{String, Nothing}="ROUND_EVEN")
        scale = 2^fraction;
        range = 2^bits;
        if min_int === nothing
            min = unsigned ? 0 : (-range//2); 
        else
            min = min_int;
        end
        if max_int === nothing
            max = unsigned ? range -1 : (range//2) -1;
        else
            max = max_int; 
        end
        new(bits, fraction, min, max, unsigned, range, scale, ovflw_behav, undflw_behav);
    end
end

"""
```
struct Fixpoint
    data :: Array{Integer}
    scheme :: FixpointScheme
```
Fixpoint type that accepts an integer array or single integer accompanied by a FixpointScheme that
governs it's handling.

See also: [`FixpointScheme`](@ref)
"""
struct Fixpoint
    data :: Array{Integer}
    scheme :: FixpointScheme
    Fixpoint(fx_data::Array{<:Integer}, scheme::FixpointScheme) = new(fx_data,scheme);
    Fixpoint(fx_data::Integer,scheme::FixpointScheme) = new([fx_data],scheme);
end

#########################################################################################
# Float parsing funtions
#########################################################################################
"""
```
fromFloat(fl_data :: Array{<:Real}, scheme :: FixpointScheme)
```
Converts a floating point array to Fixpoint according to the FixpointScheme presented.

See also: [`toFloat`](@ref)
"""
function fromFloat(fl_data::Array{<:Real}, scheme::FixpointScheme) 
    prod = fl_data .* scheme.scale;
    rnd_behav = RoundNearest;
    if (scheme.undflw_behav == "ROUND_EVEN")
        rnd_behav = RoundNearest;
    elseif (scheme.undflw_behav =="ROUND_AWAY")
        rnd_behav = RoundNearestTiesAway;
    elseif (scheme.undflw_behav =="TRUNCATE")
        rnd_behav = RoundToZero;
    else
        error("No recognisable rounding method specified");
    end
    if (scheme.ovflw_behav == "SATURATE")
        data = clamp.(round.(Integer,prod, rnd_behav),scheme.min,scheme.max);
    elseif (scheme.ovflw_behav == "WRAP")
        data = clamp_wrap.(round.(Integer,prod, rnd_behav),scheme.min,scheme.max);
    else 
        error("No recognisable overflow method specified");
    end

    return Fixpoint(data,scheme);
end

"""
```
toFloat(f :: Fixpoint)
```
Converts a Fixpoint array to floating point according to the Fixpoint's FixpointScheme.

See also: [`fromFloat`](@ref)
"""
function toFloat(f :: Fixpoint)
    fl_val = Float64.(f.data)./f.scheme.scale;
    return length(fl_val) == 1 ? fl_val[1] : fl_val;
end

# function normalise(f :: Fixpoint, min :: Integer, max :: Integer)

# end

#######################################################################################
# Arithmetic functions
#######################################################################################

"""
```
sum(f :: Fixpoint, dims :: Union{Integer,Colon})
```
Overload sum function to take Fixpoint type array as argument.

See also: [`sum`](@ref)
"""
function sum(f :: Fixpoint; dims :: Union{Integer,Colon}=:)
    sum_val = sum(f.data, dims=dims);
    bits = f.scheme.bits + ceil.(Integer,log2.(length(f.data)/length(sum_val)));
    scheme = FixpointScheme(bits, f.scheme.fraction, min_int=f.scheme.min,
    max_int=f.scheme.max, unsigned=f.scheme.unsigned, ovflw_behav=f.scheme.ovflw_behav,
    undflw_behav=f.scheme.undflw_behav);
    return Fixpoint(sum_val,scheme);
end

"""
```
*(a :: Fixpoint, b :: Fixpoint)
```
Overload * function to take Fixpoint type arrays as arguments.
        
See also: [`*`](@ref)
"""
function *(a :: Fixpoint, b :: Fixpoint)
    prod_val = a.data .* b.data;
    bits = a.scheme.bits + b.scheme.bits;
    fraction = a.scheme.fraction + b.scheme.fraction;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    return Fixpoint(prod_val,scheme);
end

"""
```
+(a :: Fixpoint, b :: Fixpoint)
```

Overload + function to take Fixpoint type arrays as arguments.
        
See also: [`+`](@ref)
"""
function +(a :: Fixpoint, b :: Fixpoint)
    if (a.scheme.scale != b.scheme.scale)
        error("Addition performed between two Fixpoint values of differing scales.");
    end
    add_val = a.data .+ b.data;
    bits = max(a.scheme.bits,b.scheme.bits) + 1;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, a.scheme.fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    return Fixpoint(add_val,scheme);
end

"""
```
-(a :: Fixpoint, b :: Fixpoint)
```

Overload - function to take Fixpoint type arrays as arguments.
        
See also: [`-`](@ref)
"""
function -(a :: Fixpoint, b :: Fixpoint)
    if (a.scheme.scale != b.scheme.scale)
        error("Subtraction performed between two Fixpoint values of differing scales.");
    end 
    sub_val = a.data .- b.data;
    bits = max(a.scheme.bits,b.scheme.bits) + 1;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, a.scheme.fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    return Fixpoint(sub_val,scheme);
end

#######################################################################################
# Misc data handling functions
#######################################################################################

"""
```
clamp_wrap(f :: Fixpoint, min :: Integer, max :: Integer)
```
An overload of clamp_wrap to take a Fixpoint array argument instead of an Integer.
        
See also: [`clamp_wrap`](@ref)
"""
function clamp_wrap(f :: Fixpoint, min :: Integer, max :: Integer)
    clamp_val = ((f.data .- min) .% (min - max)) .+ min;
    scheme = FixpointScheme(f.scheme.bits,f.scheme.fraction,unsigned=f.scheme.unsigned,
    max_int=max, min_int=min,
    ovflw_behav=f.scheme.ovflw_behav, undflw_behav=f.scheme.undflw_behav);
    return Fixpoint(clamp_val,scheme);        
end

"""
```
clamp_wrap(f :: Integer, min :: Integer, max :: Integer)
```
Does a clamp operation but wraps the value to min/max rather than saturate 
the value like standard clamp.
        
See also: [`clamp`](@ref)
"""
function clamp_wrap(i :: Integer, min :: Integer, max :: Integer)
    return ((i - min) % (max - min)) + min;
end
