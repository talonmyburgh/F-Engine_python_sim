import Base: sum, +, -, *, <<, >>, typemin, typemax, show, copy, getindex, setindex!,
        size, zeros, hcat, lastindex, axes
using Printf

#########################################################################################
# Fixpoint Structures
#########################################################################################
"""
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
Fixpoint type that accepts an integer array or single integer accompanied by a FixpointScheme that
governs it's handling.

See also: [`FixpointScheme`](@ref)
"""
struct Fixpoint{N}
    data :: Array{<:Integer,N}
    scheme :: FixpointScheme
end
Fixpoint{1}(fx_data::Integer, scheme) = Fixpoint{1}([fx_data],scheme);

"""
CFixpoint is the complex extension of Fixpoint that holds two Fixpoint types (real and imag) as its 
Real and Imaginary parts.

See also: [`Fixpoint`](@ref)
"""
struct CFixpoint{N}
    real :: Fixpoint{N}
    imag :: Fixpoint{N}
    function CFixpoint{N}(real :: Array{<:Integer,N}, imag :: Array{<:Integer,N}, scheme :: FixpointScheme) where {N}
        return new(Fixpoint{N}(real,scheme),Fixpoint{N}(imag,scheme));
    end
    function CFixpoint{N}(real :: Fixpoint{N}, imag :: Fixpoint{N}) where {N}
        if real.scheme != imag.scheme
            error("Real and Imag Fixpoint values must have the same scheme.");
        else
            return new(real,imag);
        end
    end
end

"""
Make use of Linear indexing rather than cartesian to allow for overloading of array functions
"""
Base.IndexStyle(::Fixpoint) = IndexLinear();
Base.IndexStyle(::CFixpoint) = IndexLinear();


#########################################################################################
# Float parsing funtions
#########################################################################################
"""
Converts a floating point value to Fixpoint according to the FixpointScheme presented.

See also: [`toFloat`](@ref)
"""
function fromFloat(fl_data::Real,scheme::FixpointScheme) :: Fixpoint{1}
    return fromFloat([fl_data],scheme);
end
"""
Converts a floating point array to Fixpoint according to the FixpointScheme presented.

See also: [`toFloat`](@ref)
"""
function fromFloat(fl_data::Array{<:Real,N}, scheme::FixpointScheme) :: Fixpoint{N} where {N}
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
    return Fixpoint{N}(data,scheme);
end

"""
Converts a Fixpoint array to floating point according to the Fixpoint's FixpointScheme.

See also: [`fromFloat`](@ref)
"""
function toFloat(f :: Fixpoint{N}) :: Array{Float64,N} where {N}
    fl_val = Float64.(f.data)./f.scheme.scale;
    return fl_val;
end

"""
Converts a Complex value to CFixpoint according to the FixpointScheme presented.

See also: [`toComplex`](@ref)
"""
function fromComplex(c_data::Complex, scheme::FixpointScheme) :: CFixpoint{1}
    return CFixpoint{1}(fromFloat(real(c_data),scheme), fromFloat(imag(c_data),scheme));
end
"""
Converts a Complex value to CFixpoint according to the FixpointScheme presented.

See also: [`toComplex`](@ref)
"""
function fromComplex(c_data::Array{<:Complex,N}, scheme::FixpointScheme) :: CFixpoint{N} where {N}
    return CFixpoint{N}(fromFloat(real(c_data),scheme), fromFloat(imag(c_data),scheme));
end

"""
Converts two floating point values to CFixpoint according to the FixpointScheme presented.

See also: [`toComplex`](@ref)
"""
function fromComplex(r_data::Real, i_data::Real, scheme::FixpointScheme) :: CFixpoint{1}
    return CFixpoint{1}(fromFloat(r_data,scheme),fromFloat(i_data,scheme));
end

"""
Converts two floating point values to CFixpoint according to the FixpointScheme presented.

See also: [`toComplex`](@ref)
"""
function fromComplex(r_data::Array{<:Real,N}, i_data::Array{<:Real,N}, scheme::FixpointScheme) :: CFixpoint{N} where {N}
    return CFixpoint{N}(fromFloat(r_data,scheme),fromFloat(i_data,scheme));
end

"""
Converts a CFixpoint array to a complex point array according to the CFixpoint's FixpointScheme.
"""
function toComplex(cfix :: CFixpoint{N}) :: Array{ComplexF64,N} where {N}
    return toFloat(cfix.real) + toFloat(cfix.imag)*im;
end

"""
Creates a Fixpoint populated with zero values. Creates a CFixpoint if complex is set to true.
"""
function zeros(fx_scheme :: FixpointScheme, dims :: NTuple{N,Int}; complex :: Bool = false) :: Union{Fixpoint{N},CFixpoint{N}} where{N}
    if complex
        return fromComplex(zeros(Float64, dims),zeros(Float64, dims),fx_scheme);
    else
        return fromFloat(zeros(Float64, dims), fx_scheme); 
    end
end

"""
Fit all data within the min/max values for Fixpoint
"""
function normalise(f_val :: Fixpoint{N}) :: Fixpoint{N} where {N}
    return Fixpoint{N}(clamp.(f_val.data,f_val.scheme.min,f_val.scheme.max),f_val.scheme);
end

"""
Fit all data within the min/max values for CFixpoint
"""
function normalise(cf_val :: CFixpoint{N}) :: CFixpoint{N} where {N}
    return CFixpoint{N}(normalise(cf_val.real),normalise(cf_val.imag));
end

"""
Cast Fixpoint type to Fixpoint type of new scheme
"""
function cast(f_val :: Fixpoint{N}, f_scheme :: FixpointScheme) :: Fixpoint{N} where {N}
    return Fixpoint{N}(f_val.data,f_scheme);
end

"""
Cast CFixpoint type to CFixpoint type of new scheme
"""
function cast(cf_val :: CFixpoint{N}, f_scheme :: FixpointScheme) :: CFixpoint{N} where {N}
    return CFixpoint{N}(Fixpoint(cf_val.real.data,f_scheme),Fixpoint(cf_val.imag.data,f_scheme));
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
function sum(f :: Fixpoint{N}; dims :: Union{Integer,Colon}=:) :: Fixpoint{N} where {N}
    sum_val = sum(f.data, dims=dims);
    bits = f.scheme.bits + ceil.(Integer,log2.(length(f.data)/length(sum_val)));
    scheme = FixpointScheme(bits, f.scheme.fraction, min_int=f.scheme.min,
    max_int=f.scheme.max, unsigned=f.scheme.unsigned, ovflw_behav=f.scheme.ovflw_behav,
    undflw_behav=f.scheme.undflw_behav);
    return Fixpoint{N}(sum_val,scheme);
end

"""
```
sum(cf :: CFixpoint, dims :: Union{Integer,Colon}=:)
```
Overload sum function to take CFixpoint type array as argument.
See also: [`sum`](@ref)
"""
function sum(cf :: CFixpoint{N}; dims :: Union{Integer,Colon}=:) :: CFixpoint{N} where {N}
    r_sum_val = sum(cf.real,dims=dims);
    i_sum_val = sum(cf.imag,dims=dims);
    return CFixpoint{N}(r_sum_val,i_sum_val);
end

"""
```
*(a :: Fixpoint, b :: Fixpoint)
```
Overload * function to take Fixpoint type arrays as arguments.
        
See also: [`*`](@ref)
"""
function *(a :: Fixpoint{N}, b :: Fixpoint{N}) :: Fixpoint{N} where {N}
    prod_val = a.data .* b.data;
    bits = a.scheme.bits + b.scheme.bits;
    fraction = a.scheme.fraction + b.scheme.fraction;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    return Fixpoint{N}(prod_val,scheme);
end

"""
```
*(a :: CFixpoint, b :: CFixpoint) :: CFixpoint
```
Overload * function to take CFixpoint type arrays as arguments.

See also: [`*`](@ref)
"""
function *(a :: CFixpoint{N}, b :: CFixpoint{N}) :: CFixpoint{N} where {N}
    function cmult(a, b, c, d)
        # Real part x = a*c - b*d
        x = (a*c)-(b*d);
        # Imaginary part y = a*d + b*c
        y = (a*d)+(b*c);
        return x, y;
    end
    out_real, out_imag = cmult(a.real, a.imag, b.real, b.imag);
    return CFixpoint{N}(out_real, out_imag); 
end

"""
```
+(a :: Fixpoint, b :: Fixpoint)
```
Overload + function to take Fixpoint type arrays as arguments.
        
See also: [`+`](@ref)
"""
function +(a :: Fixpoint{N}, b :: Fixpoint{N}) :: Fixpoint{N} where {N}
    if (a.scheme.scale != b.scheme.scale)
        error("Addition performed between two Fixpoint values of differing scales.");
    end
    add_val = a.data .+ b.data;
    bits = max(a.scheme.bits,b.scheme.bits) + 1;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, a.scheme.fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    return Fixpoint{N}(add_val,scheme);
end
"""
```
+(a :: CFixpoint, b :: CFixpoint)
```
Overload + function to take CFixpoint type arrays as arguments.
        
See also: [`+`](@ref)
"""
function +(a :: CFixpoint{N}, b :: CFixpoint{N}) :: CFixpoint{N} where {N}
    r_sum = a.real + b.real;
    i_sum = a.imag + b.imag;
    return CFixpoint{N}(r_sum, i_sum);
end

"""
```
-(a :: Fixpoint, b :: Fixpoint)
```
Overload - function to take Fixpoint type arrays as arguments.
        
See also: [`-`](@ref)
"""
function -(a :: Fixpoint{N}, b :: Fixpoint{N}) :: Fixpoint{N} where {N}
    if (a.scheme.scale != b.scheme.scale)
        error("Subtraction performed between two Fixpoint values of differing scales.");
    end 
    sub_val = a.data .- b.data;
    bits = max(a.scheme.bits,b.scheme.bits) + 1;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, a.scheme.fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    return Fixpoint{N}(sub_val,scheme);
end

"""
```
-(a :: CFixpoint, b :: CFixpoint)
```
Overload - function to take CFixpoint type arrays as arguments.
        
See also: [`-`](@ref)
"""
function -(a :: CFixpoint{N}, b :: CFixpoint{N}) :: CFixpoint{N} where {N}
    r_sub = a.real - b.real;
    i_sub = a.imag - b.imag;
    return CFixpoint{N}(r_sub, i_sub);
end

"""
```
power(f :: Fixpoint)
```
Returns power of the Fixpoint value given = f.data * f.data.
"""
function power(f :: Fixpoint{N}) :: Array{Integer,N} where {N}
    return f.data .* f.data;
end

"""
```
power(cf :: CFixpoint)
```
Returns power of the CFixpoint value given = cf * conj(cf).
See also: [`conj`](@ref)
"""
function power(cf :: CFixpoint{N}) :: Array{Integer, N} where {N}
    res = copy(cf) * conj(cf);
    return res.real;
end

"""
```
conj(cf :: CFixpoint)
```
Returns conjuage of the CFixpoint value given.
"""
function conj(cf :: CFixpoint{N}) :: CFixpoint{N} where {N}
    i_res = copy(cf.imag);
    i_res.data = - copy(cf.imag.data);
    return CFixpoint{N}(cf.real, i_res);
end

"""
```
conj!(cf :: CFixpoint)
```
Returns conjuage of the CFixpoint value given.
! implies inline operation.
"""
function conj(cf :: CFixpoint{N}) :: CFixpoint{N} where {N}
    cf.imag.data = -cf.imag.data;
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
function clamp_wrap(f :: Fixpoint{N}, min :: Integer, max :: Integer) :: Fixpoint{N} where {N}
    clamp_val = ((f.data .- min) .% (min - max)) .+ min;
    scheme = FixpointScheme(f.scheme.bits,f.scheme.fraction,unsigned=f.scheme.unsigned,
    max_int=max, min_int=min,
    ovflw_behav=f.scheme.ovflw_behav, undflw_behav=f.scheme.undflw_behav);
    return Fixpoint{N}(clamp_val,scheme);        
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
Requantise the data contained in fxpt according to the new scheme provided.
"""
function quantise(fxpt :: Fixpoint{N}, scheme :: FixpointScheme) :: Fixpoint{N} where {N}
    return fromFloat(toFloat(fxpt), scheme);
end

"""
Requantise the data contained in cfxpt according to the new scheme provided.
"""
function quantise(cfxpt :: CFixpoint{N}, scheme :: FixpointScheme) :: CFixpoint{N} where {N}
    return fromComplex(toComplex(cfxpt),scheme);
end

"""
Overload copy() function to copy Fixpoint by value as opposed to reference.

See also: [`copy`](@ref)
"""
function copy(f :: Fixpoint{N}) :: Fixpoint{N} where {N}
    tmpscheme = FixpointScheme(f.scheme.bits, f.scheme.fraction, min_int=f.scheme.min,
    max_int=f.scheme.max, unsigned=f.scheme.unsigned, ovflw_behav=f.scheme.ovflw_behav,
    undflw_behav=f.scheme.undflw_behav);
    return Fixpoint{N}(copy(f.data),tmpscheme);
end

"""
```
function copy(cf :: CFixpoint)
```
Overload copy() function to copy CFixpoint by value as opposed to reference.
"""
function copy(cf :: CFixpoint{N}) :: CFixpoint{N} where {N}
    return CFixpoint{N}(copy(cf.real),copy(cf.imag));
end

"""
```
function size(f :: Fixpoint)
```
Overload size() function to accept Fixpoint.
"""
function size(f::Fixpoint{N}) where {N}
    return size(f.data);
end

"""
```
function size(cf :: CFixpoint)
```
Overload size() function to accept CFixpoint.
"""
function size(cf::CFixpoint{N}) where {N}
    return size(cf.real);
end


"""
```
show(io :: IO, f :: Fixpoint)
```
Overload show function for printing out Fixpoint summary.
See also: [`show`](@ref)
"""
function show(io::IO, f :: Fixpoint{N}) where {N}
    @printf(io,"Fixpoint real %s (%d, %d), shape %s", f.scheme.unsigned ? "unsigned" : "signed",f.scheme.bits, f.scheme.fraction, size(f.data));
end

"""
```
show(io :: IO, f :: Fixpoint)
```
Overload show function for printing out Fixpoint summary.
See also: [`show`](@ref)
"""
function show(io::IO, cf :: CFixpoint{N}) where {N}
    @printf(io,"CFixpoint complex %s (%d, %d), shape %s", cf.real.scheme.unsigned ? "unsigned" : "signed",cf.real.scheme.bits, cf.real.scheme.fraction, size(cf.real.data))
end

"""
```
getindex(f :: Fixpoint, i :: Int)
```
Overload getindex function for accessing data elements out Fixpoint type.
"""
function getindex(f :: Fixpoint{1}, i :: Int) :: Fixpoint{1} 
    return Fixpoint{1}(getindex(f.data,i),f.scheme);
end

"""
```
getindex(cf :: CFixpoint, i :: Int)
```
Overload getindex function for accessing data elements out CFixpoint type.
"""
function getindex(cf :: CFixpoint{1}, i :: Int) :: CFixpoint{1}
    return CFixpoint{1}(getindex(cf.real,i),getindex(cf.imag,i));
end

"""
```
getindex(f :: Fixpoint, I::Vararg{Int, N})
```
Overload getindex function for accessing data elements out Fixpoint type.
"""
function getindex(f :: Fixpoint{N}, I::Vararg{Int, N}) :: Fixpoint{N} where {N}
    return Fixpoint{N}(f.data[I...],f.scheme);
end

"""
```
getindex(cf :: CFixpoint, I::Vararg{Int, N})
```
Overload getindex function for accessing data elements out CFixpoint type.
"""
function getindex(cf :: CFixpoint{N}, I::Vararg{Int, N}) :: CFixpoint{N}  where {N}
    return CFixpoint{N}(cf.real[I...],cf.imag[I...]);
end

"""
Overload getindex function for accessing data elements out Fixpoint type.
"""
function getindex(f :: Fixpoint{N}, i :: UnitRange{Int64}) :: Fixpoint{N} where {N}
    return Fixpoint{N}(f.data[i],f.scheme);
end

"""
Overload getindex function for accessing data elements out CFixpoint type.
"""
function getindex(cf :: CFixpoint, i :: UnitRange{Int64}) :: CFixpoint
    return CFixpoint(cf.real[i],cf.imag[i]);
end

"""
Overload getindex function for accessing data elements out Fixpoint type.
"""
function getindex(f :: Fixpoint{1}, i :: Vector{Int}) :: Fixpoint{1}
    return Fixpoint{1}(f.data[i],f.scheme);
end

"""
Overload getindex function for accessing data elements out CFixpoint type.
"""
function getindex(cf :: CFixpoint{1}, i :: Vector{Int}) :: CFixpoint{1}
    return CFixpoint{1}(cf.real[i],cf.imag[i]);
end

"""
Overload setindex! function for accessing data elements out Fixpoint type.
"""
function setindex!(f :: Fixpoint{N}, val :: Fixpoint{1}, i :: Int) :: Nothing where {N}
    f.data[i] = val.data;
end

"""
Overload setindex! function for accessing data elements out CFixpoint type.
"""
function setindex!(cf :: CFixpoint{N}, val :: CFixpoint{1}, i :: Int) :: Nothing where {N}
    cf.real[i] = val.real;
    cf.imag[i] = val.imag;
end

"""
Overload setindex function for accessing data elements out Fixpoint type.
"""
function setindex!(f :: Fixpoint{N}, val :: Fixpoint{M}, I::Vararg{Int,N}) :: Nothing where {N,M}
    f.data[I] = val.data;
end

"""
Overload setindex function for accessing data elements out CFixpoint type.
"""
function setindex!(cf :: CFixpoint{N}, val :: CFixpoint{M}, I::Vararg{Int,N}) :: Nothing where {N,M}
    cf.real[I] = val.real;
    cf.imag[I] = val.imag;
end

# """
# ```
# setindex!(f :: Fixpoint, i ::UnitRange{Int})
# ```
# Overload setindex function for accessing data elements out Fixpoint type.
# """
# function setindex!(f :: Fixpoint, val :: Fixpoint, i :: UnitRange{Int}) :: Nothing
#     f.data[i] = val.data;
#     return;
# end

# """
# ```
# setindex!(cf :: CFixpoint, i ::UnitRange{Int})
# ```
# Overload setindex function for accessing data elements out CFixpoint type.
# """
# function setindex!(cf :: CFixpoint, val :: CFixpoint, i :: UnitRange{Int}) :: Nothing
#     cf.real[i] = val.real;
#     cf.imag[i] = val.imag;
#     return;
# end



# """
# Overload lastindex function for Fixpoint functions
# """
# function lastindex(fx::Fixpoint) :: Integer
#     return lastindex(fx.data);
# end

# """
# Overload lastindex function for CFixpoint functions
# """
# function lastindex(cfx::CFixpoint) :: Integer
#     return lastindex(cfx.real.data);
# end

# """
# Overload axes function for Fixpoint function
# """
# function axes(fx  :: Fixpoint, i :: Integer) :: AbstractUnitRange{<:Integer}
#     return axes(fx.data, i);
# end

# """
# Overload axes function for CFixpoint function
# """
# function axes(cfx  :: CFixpoint, i :: Integer) :: AbstractUnitRange{<:Integer}
#     return axes(cfx.real.data, i);
# end

"""
Overload hcat function to handle horizontal concatenation of Fixpoint types.
Requires that schemes match.
"""
function hcat(f_1 :: Fixpoint{N}, f_2 :: Fixpoint{N}) :: Fixpoint{N} where {N}
    #Check schemes match:
    if f_1.scheme == f_2.scheme
        return Fixpoint{N}(hcat(f_1.data,f_2.data),f_1.scheme);
    else
        error("Fixpoint args don't share the same scheme.");
    end
end

"""
Overload hcat function to handle horizontal concatenation of CFixpoint types.
Requires that schemes match.
"""
function hcat(cf_1 :: CFixpoint{N}, cf_2 :: CFixpoint{N}) :: CFixpoint{N} where {N}
    #Check real schemes match - imag will match:
    if cf_1.real.scheme == cf_2.real.scheme
        return CFixpoint{N}(hcat(cf_1.real,cf_2.real), hcat(cf_1.imag,cf_2.imag));
    else
        error("CFixpoint args don't share the same scheme.");
    end
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
function >>(fxpt :: Fixpoint{N}, steps :: Integer) :: Fixpoint{N} where {N}
    if (steps < 0)
        error("Integer value for steps must be greater than or equal to zero.");
    else
        if (fxpt.scheme.undflw_behav == "ROUND_EVEN")
            rnd_behav = RoundNearest;
        elseif (fxpt.scheme.undflw_behav =="ROUND_AWAY")
            rnd_behav = RoundNearestTiesAway;
        elseif (fxpt.scheme.undflw_behav =="TRUNCATE")
            rnd_behav = RoundToZero;
        else
            error("No recognisable rounding method specified");
        end
        return Fixpoint{N}(round.(Integer, fxpt.data/(2^steps),rnd_behav),fxpt.scheme);
    end
end

"""
```
>>(cfxpt :: CFixpoint, steps :: Integer)
```
Overload >> function for CFixpoint args.
Apply 'steps' (>=0) right shifts to cfxpt. Cannot use >> operator here since we must control rounding.

See also: [`>>`](@ref)
"""
function >>(cfxpt :: CFixpoint{N}, steps :: Integer) :: CFixpoint{N} where {N}
    t_real = cfxpt.real >> steps;
    t_imag = cfxpt.imag >> steps;
    return CFixpoint{N}(t_real, t_imag);
end

"""
```
<<(fxpt :: Fixpoint, steps :: Integer)
```
Overload << function for Fixpoint args.
Apply 'steps' (>=0) left shifts to fxpt.
See also: [`<<`](@ref)
"""
function <<(fxpt :: Fixpoint{N}, steps :: Integer) :: Fixpoint{N} where {N}
    t_fxpt = copy(fxpt);
    if (steps < 0)
        error("Integer value for steps must be greater than or equal to zero.");
    else    
        t_fxpt.data .<<= steps;
    end
    return t_fxpt;
end

"""
```
<<(fxpt :: Fixpoint, steps :: Integer)
```
Overload << function for Fixpoint args.
Apply 'steps' (>=0) left shifts to fxpt.
See also: [`<<`](@ref)
"""
function <<(cfxpt :: CFixpoint{N}, steps :: Integer) :: CFixpoint{N} where {N}
    t_real = cfxpt.real << steps;
    t_imag = cfxpt.imag << steps;
    return CFixpoint{N}(t_real,t_imag);
end