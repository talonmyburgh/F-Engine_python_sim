import Base: sum, +, -, *, typemin, typemax, show, copy
using Printf

#########################################################################################
# Fixpoint Structures
#########################################################################################
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

"""
```
struct CFixpoint
    real :: Fixpoint
    imag :: Fixpoint
```
CFixpoint is the complex extension of Fixpoint that holds two Fixpoint types (real and imag) as its 
Real and Imaginary parts.

See also: [`Fixpoint`](@ref)
"""
struct CFixpoint
    real :: Fixpoint
    imag :: Fixpoint
    function CFixpoint(real :: Array{<:Integer}, imag :: Array{<:Integer}, scheme :: FixpointScheme)
        return new(Fixpoint(real,scheme),Fixpoint(imag,scheme));
    end
    function CFixpoint(real :: Fixpoint, imag :: Fixpoint)
        if real.scheme != imag.scheme
            error("Real and Imag Fixpoint values must have the same scheme.");
        else
            return new(real,imag);
        end
    end
end

#########################################################################################
# Float parsing funtions
#########################################################################################
"""
```
fromFloat(fl_data::Real,scheme::FixpointScheme)
```
Converts a floating point value to Fixpoint according to the FixpointScheme presented.

See also: [`toFloat`](@ref)
"""
function fromFloat(fl_data::Real,scheme::FixpointScheme) :: Fixpoint
    return fromFloat([fl_data],scheme);
end
"""
```
fromFloat(fl_data :: Array{<:Real}, scheme :: FixpointScheme)
```
Converts a floating point array to Fixpoint according to the FixpointScheme presented.

See also: [`toFloat`](@ref)
"""
function fromFloat(fl_data::Array{<:Real}, scheme::FixpointScheme) :: Fixpoint
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
function toFloat(f :: Fixpoint) :: Array{Float64}
    fl_val = Float64.(f.data)./f.scheme.scale;
    return length(fl_val) == 1 ? fl_val[1] : fl_val;
end

"""
```
fromComplex(c_data::Array{<:Complex}, scheme::FixpointScheme)
```
Converts a Complex value to CFixpoint according to the FixpointScheme presented.

See also: [`toComplex`](@ref)
"""
function fromComplex(c_data::Union{Array{<:Complex},Complex}, scheme::FixpointScheme) :: CFixpoint
    return CFixpoint(fromFloat(real(c_data),scheme), fromFloat(imag(c_data),scheme));
end

"""
```
fromComplex(r_data::Union{Array{<:Real},Real}, i_data::Union{Array{<:Real},Real}, scheme::FixpointScheme)
```
Converts a two floating point values to CFixpoint according to the FixpointScheme presented.

See also: [`toComplex`](@ref)
"""
function fromComplex(r_data::Union{Array{<:Real},Real}, i_data::Union{Array{<:Real},Real}, scheme::FixpointScheme) :: CFixpoint
    return CFixpoint(fromFloat(r_data,scheme),fromFloat(i_data,scheme));
end

"""
```
toComplex(cfix::CFixpoint)
```
Converts a CFixpoint array to a complex point array according to the CFixpoint's FixpointScheme.
"""
function toComplex(cfix :: CFixpoint) :: Array{ComplexF64}
    return toFloat(cfix.real) + toFloat(cfix.imag)*im;
end

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
function sum(f :: Fixpoint; dims :: Union{Integer,Colon}=:) :: Fixpoint
    sum_val = sum(f.data, dims=dims);
    bits = f.scheme.bits + ceil.(Integer,log2.(length(f.data)/length(sum_val)));
    scheme = FixpointScheme(bits, f.scheme.fraction, min_int=f.scheme.min,
    max_int=f.scheme.max, unsigned=f.scheme.unsigned, ovflw_behav=f.scheme.ovflw_behav,
    undflw_behav=f.scheme.undflw_behav);
    return Fixpoint(sum_val,scheme);
end

"""
```
sum(cf :: CFixpoint, dims :: Union{Integer,Colon}=:)
```
Overload sum function to take CFixpoint type array as argument.
See also: [`sum`](@ref)
"""
function sum(cf :: CFixpoint; dims :: Union{Integer,Colon}=:) :: CFixpoint
    r_sum_val = sum(cf.real,dims=dims);
    i_sum_val = sum(cf.imag,dims=dims);
    return CFixpoint(r_sum_val,i_sum_val,r_sum_val.scheme);
end

"""
```
*(a :: Fixpoint, b :: Fixpoint)
```
Overload * function to take Fixpoint type arrays as arguments.
        
See also: [`*`](@ref)
"""
function *(a :: Fixpoint, b :: Fixpoint) :: Fixpoint
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
*(a :: CFixpoint, b :: CFixpoint) :: CFixpoint
```
Overload * function to take CFixpoint type arrays as arguments.

See also: [`*`](@ref)
"""
function *(a :: CFixpoint, b :: CFixpoint) :: CFixpoint
    function cmult(a, b, c, d)
        # Real part x = a*c - b*d
        x = (a*c)-(b*d);
        # Imaginary part y = a*d + b*c
        y = (a*d)+(b*c);
        return x, y;
    end
    out_real, out_imag = cmult(a.real, a.imag, b.real, b. imag);
    return CFixpoint(out_real, out_imag, out_real.scheme); 
end

"""
```
+(a :: Fixpoint, b :: Fixpoint)
```
Overload + function to take Fixpoint type arrays as arguments.
        
See also: [`+`](@ref)
"""
function +(a :: Fixpoint, b :: Fixpoint) :: Fixpoint
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
+(a :: CFixpoint, b :: CFixpoint)
```
Overload + function to take CFixpoint type arrays as arguments.
        
See also: [`+`](@ref)
"""
function +(a :: CFixpoint, b :: CFixpoint) :: CFixpoint
    r_sum = a.real + b.real;
    i_sum = a.imag + b.imag;
    return CFixpoint(r_sum, i_sum, r_sum.scheme);
end

"""
```
-(a :: Fixpoint, b :: Fixpoint)
```
Overload - function to take Fixpoint type arrays as arguments.
        
See also: [`-`](@ref)
"""
function -(a :: Fixpoint, b :: Fixpoint) :: Fixpoint
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

"""
```
-(a :: CFixpoint, b :: CFixpoint)
```
Overload - function to take CFixpoint type arrays as arguments.
        
See also: [`-`](@ref)
"""
function -(a :: CFixpoint, b :: CFixpoint) :: CFixpoint
    r_sub = a.real - b.real;
    i_sub = a.imag - b.imag;
    return CFixpoint(r_sub, i_sub, r_sub.scheme);
end

"""
```
power(f :: Fixpoint)
```
Returns power of the Fixpoint value given = f.data * f.data.
"""
function power(f :: Fixpoint) :: Array{Integer}
    return f.data .* f.data;
end

"""
```
power(cf :: CFixpoint)
```
Returns power of the CFixpoint value given = cf * conj(cf).
See also: [`conj`](@ref)
"""
function power(cf :: CFixpoint) :: Array{Integer}
    res = copy(cf) * conj(cf);
    return res.real;
end

"""
```
conj(cf :: CFixpoint)
```
Returns conjuage of the CFixpoint value given.
"""
function conj(cf :: CFixpoint) :: CFixpoint
    i_res = copy(cf.imag);
    i_res.data = - copy(cf.imag.data);
    return CFixpoint(cf.real, i_res);
end

#######################################################################################
# Misc Fixpoint type handling functions
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

"""
```
quantise(fxpt :: Fixpoint, scheme :: FixpointScheme)
```
Requantise the data contained in fxpt according to the new scheme provided.
"""
function quantise(fxpt :: Fixpoint, scheme :: FixpointScheme)
    return fromFloat(toFloat(fxpt), scheme);
end

"""
```
copy(f :: Fixpoint)
```
Overload copy() function to copy Fixpoint by value as opposed to reference.

See also: [`copy`](@ref)
"""
function copy(f :: Fixpoint)
    tmpscheme = FixpointScheme(f.scheme.bits, f.scheme.fraction, min_int=f.scheme.min,
    max_int=f.scheme.max, unsigned=f.scheme.unsigned, ovflw_behav=f.scheme.ovflw_behav,
    undflw_behav=f.scheme.undflw_behav);
    return Fixpoint(copy(f.data),tmpscheme);
end


"""
```
show(io :: IO, f :: Fixpoint)
```
Overload show function for printing out Fixpoint summary.
See also: [`show`](@ref)
"""
function show(io::IO, f :: Fixpoint)
    @printf(io,"Fixpoint real %s (%d, %d), shape %s", f.scheme.unsigned ? "unsigned" : "signed",f.scheme.bits, f.scheme.fraction, size(f.data));
end

#######################################################################################
# Logical operator functions
#######################################################################################
"""
```
>>(fxpt :: Fixpoint, steps :: Integer)
```
Overload >> function for Fixpoint args.
Apply 'steps' (>=0) right shifts to fxpt. Cannot use >> operator here since we must control rounding.

See also: [`>>`](@ref)
"""
function >>(fxpt :: Fixpoint, steps :: Integer)
    if (steps < 0)
        error("Integer value for steps must be greater than or equal to zero.");
    else
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
        fxpt.data = round.(Integer, fxpt.data/(2^steps),rnd_behav);
    end
    return fxpt;
end

"""
```
<<(fxpt :: Fixpoint, steps :: Integer)
```
Overload << function for Fixpoint args.
Apply 'steps' (>=0) left shifts to fxpt. Make use of << operator here.

See also: [`<<`](@ref)
"""
function <<(fxpt :: Fixpoint, steps :: Integer)
    if (steps < 0)
        error("Integer value for steps must be greater than or equal to zero.");
    else    
        fxpt.data <<= steps;
    end
    return fxpt;
end