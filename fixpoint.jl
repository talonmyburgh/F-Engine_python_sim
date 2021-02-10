import Base: sum, +, -, *, typemin, typemax

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

struct Fixpoint
    data :: Array{Integer}
    scheme :: FixpointScheme
    function Fixpoint(fx_data::Array{<:Integer}, scheme::FixpointScheme)
        new(fx_data,scheme);
    end
end

function fromFloat(fl_data::Array{<:Real}, scheme::FixpointScheme) 
    prod = fl_data .* scheme.scale;
    rnd_behav = RoundNearest;
    if (scheme.undflw_behav == "ROUND_EVEN")
        rnd_behav = RoundNearest;
    elseif (scheme.undflw_behav =="ROUND_AWAY")
        rnd_behav = RoundNearestTiesAway;
    elseif (scheme.undflw_behav =="TRUNCATE")
        rnd_behav = RoundToZero;
    end
    data = clamp.(round.(Integer,prod, rnd_behav),scheme.min,scheme.max);
    fx=Fixpoint(data,scheme);
end

function toFloat(f :: Fixpoint)
    float_val = Float64.(f.data)./f.scheme.scale;
end

# function normalise(f :: Fixpoint, min :: Integer, max :: Integer)

# end

function sum(f :: Fixpoint; dims :: Union{Integer,Nothing}=nothing)
    sum_val = sum.(f.data, dims=dims);
    bits = f.scheme.bits + Integer.(ceil.(log2.(size(Fixpoint.data)/size(sum_val))));
    scheme = FixpointScheme(bits, f.scheme.fraction, min_int=f.scheme.min_int,
    max_int=f.scheme.max_int, unsigned=f.scheme.unsigned, ovflw_behav=f.scheme.ovflw_behav,
    undflw_behav=f.scheme.undflw_behav);
    result = Fixpoint(sum_val,scheme);
end

function *(a :: Fixpoint, b :: Fixpoint)
    prod_val = a.data .* b.data;
    bits = a.scheme.bits + b.scheme.bits;
    fraction = a.scheme.fraction + b.scheme.fraction;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    result = Fixpoint(prod_val,scheme);
end

function +(a :: Fixpoint, b :: Fixpoint)
    if (a.scheme.scale>b.scheme.scale | a.scheme.scale<b.scheme.scale)
        error("Addition performed between two Fixpoint values of differing scales.");
    end
    add_val = a.data .+ b.data;
    bits = max(a.scheme.bits,b.scheme.bits) + 1;
    unsigned = a.scheme.unsigned & b.scheme.unsigned;
    scheme = FixpointScheme(bits, a.scheme.fraction, unsigned=unsigned, 
    ovflw_behav=a.scheme.ovflw_behav, undflw_behav=a.scheme.undflw_behav);
    result = Fixpoint(add_val,scheme);
end

function clamp_wrap(f :: Fixpoint, min :: Integer, max :: Integer)
    clamp_val = clamp_wrap.(f.data,min,max);
    scheme = FixpointScheme(f.scheme.bits,f.scheme.fraction,unsigned=f.scheme.unsigned,
    max_int=max, min_int=min,
    ovflw_behav=f.scheme.ovflw_behav, undflw_behav=f.scheme.undflw_behav);
    result = Fixpoint(clamp_val,scheme);        
end

function clamp_wrap(i :: Integer, min :: Integer, max :: Integer)
    if i >0
        if max >= i
            res = i;
        else
            res = min+(i-max)-1;
        end
    end
    if i<=0
        if min >i
            res = max+(i-min)+1;
        else
            res = i;
        end
    end
    return res
end
