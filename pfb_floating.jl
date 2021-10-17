import Base: conj
include("pfb_coeff_gen.jl");
"""
```
bitRevArray(array::Array{<:Complex{<:Real}}, N::Integer)::Array{<:Complex{<:Real}}
```
Bit reverses an array's contents according to bitRev of the array's indices.
Takes an array of length N which must be a power of two.
"""
function bitRevArray(array::Array{<:Complex{<:Real}}, N::Integer)::Array{<:Complex{<:Real}}
    return array[bitRev(N)];
end

"""
```
makeTwiddle(N::Integer)::Array{<:Complex{<:Real}}
```
Generates twiddle factors for the natInIterDitFFT FFT.
See also: [`natInIterDitFFT`](@ref)   
"""
function makeTwiddle(N::Integer)::Array{<:Complex{<:Real}}
    i = collect(0:div(N, 2)-1);
    twids = exp.(i .* (-2 * pi * (1im / N)));
    return twids;
end

"""
Floating point PFB implementation making use of the natural order in fft
like CASPER does.
"""
struct FloatPFBScheme
    N           :: Integer;
    dual        :: Bool;
    reg         :: Array{ComplexF64};
    staged      :: Bool;
    fwidth      :: Float64;
    chan_acc    :: Bool;
    window      :: Array{Float64};
    twids       :: Array{ComplexF64};
    function FloatPFBScheme(N::Integer, taps::Integer; w::String="hanning", dual::Bool=false, 
                            staged::Bool=false, fwidth::Float64=1.0, chan_acc::Bool=false)
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

"""
``` 
natInIterDitFFT(pfbsch::FloatPFBScheme, data::Array{<:Complex{<:Real}}) :: Array{<:Complex}
```
natInIterDitFFT accepts PFB scheme and data.
"""
function natInIterDitFFT(pfbsch::FloatPFBScheme, data::Array{<:Complex}) :: Array{<:Complex}
    data = copy(data);
    N = size(data)[1];
    if pfbsch.staged
        stgd_data = zeros(ComplexF64, N, convert(Int64, log2(N)) + 2);
        stgd_data[:,1] .= data[:];
    end
    num_of_groups = 1;
    distance = div(pfbsch.N,2);
    stg = 2;
    while num_of_groups < pfbsch.N
        for k in 0:num_of_groups-1
            jfirst = (2 * k * distance) + 1;
            jlast = jfirst + distance - 1;
            W = pfbsch.twids[k+1];
            slc1 = jfirst:jlast;
            slc2 = slc1 .+ distance;
            tmp = W .* data[slc2];
            data[slc2] .= data[slc1] .- tmp;
            data[slc1] .= data[slc1] .+ tmp;
        end
        num_of_groups *= 2;
        distance = div(distance, 2);
        if pfbsch.staged
            stgd_data[:,stg] = data[:];
        end
        stg += 1;
    end
    if pfbsch.staged
        stgd_data[:,end] .= bitRevArray(stgd_data[:,end - 1], pfbsch.N);
        return stgd_data;
    else
        return bitRevArray(data, pfbsch.N);
    end
end

"""
``` 
natInIterDitFFT(pfbsch::FloatPFBScheme, data::Array{<:Complex{<:Real}}) :: Array{<:Complex}
```
Overload natInIterDitFFT to accept a PFB scheme and two real data arrays.
"""
function natInIterDitFFT(pfbsch::FloatPFBScheme, data_a::Array{<:Complex{<:Real}}, data_b::Array{<:Complex{<:Real}}) :: Array{<:Complex}
    data = data_a .+ data_b.*im;                                                            #Merge into a complex array for processing.
    data = copy(data);
    N = size(data)[1];
    if pfbsch.staged
        stgd_data = zeros(ComplexF64, N, convert(Int64, log2(N)) + 2);
        stgd_data[:,1] .= data[:];
    end
    num_of_groups = 1;
    distance = div(pfbsch.N,2);
    stg = 2;
    while num_of_groups < pfbsch.N
        for k in 0:num_of_groups-1
            jfirst = (2 * k * distance) + 1;
            jlast = jfirst + distance - 1;
            W = pfbsch.twids[k+1];
            slc1 = jfirst:jlast;
            slc2 = slc1 .+ distance;
            tmp = W .* data[slc2];
            data[slc2] .= data[slc1] .- tmp;
            data[slc1] .= data[slc1] .+ tmp;
        end
        num_of_groups *= 2;
        distance = div(distance, 2);
        if pfbsch.staged
            stgd_data[:,stg] = data[:];
        end
        stg += 1;
    end
    if pfbsch.staged
        stgd_data[:,end] .= bitRevArray(stgd_data[:,end - 1], pfbsch.N);
        # print(stgd_data[:,end])
        return stgd_data;
    else
        return bitRevArray(data, pfbsch.N);
    end
end

"""
```
PFBFir(pfbsch::FloatPFBScheme ,x::Array{<:Complex{<:Real}}) :: Array{<:Complex{<:Real}}
```
FIR: Takes data segment (N long) and appends each value to each fir.
Returns data segment (N long) that is the sum of fircontents*windowcoeffs.
"""
function PFBFir(pfbsch::FloatPFBScheme ,x::Array{<:Complex{<:Real}}) :: Array{<:Complex{<:Real}}
    pfbsch.reg .= hcat(x,pfbsch.reg)[:,1:end-1];
    X = sum(pfbsch.reg .* pfbsch.window,dims=2);
    return X;
end

"""
```
SpecSplit(Y_k::Array{<:Complex})::Tuple{Array{<:Complex},Array{<:Complex}}
```
For dual polarisation processing, we need to split the data after
FFT and return the individual complex spectra.
"""
function SpecSplit(Y_k::Array{<:Complex})::Tuple{Array{<:Complex},Array{<:Complex}}
    R_k = real.(Y_k);
    R_kflip = copy(R_k);
    R_kflip[2:end,:].=R_kflip[end:-1:2,:];

    I_k = imag.(Y_k);
    I_kflip = copy(I_k);
    I_kflip[2:end,:].=I_kflip[end:-1:2,:];
    
    G_k = (1/2) .* (R_k .+ im .* I_k .+ R_kflip .- im .* I_kflip);
    H_k = (1/2im) .* (R_k .+ im .* I_k .- R_kflip .+ im .* I_kflip);
    return (G_k, H_k);
end

"""
```
SpecPow(pfbsch::FloatPFBScheme, X::Array{<:Complex}) :: Array{<:Real}
```
Here we take the power spectrum of the outputs. Chan_acc dictates
if one must sum over all outputs produced.
"""
function SpecPow(pfbsch::FloatPFBScheme, X::Array{<:Complex}) :: Array{<:Real}
    if pfbsch.chan_acc
        pwr = X .* conj.(X);
        pwr = real.(sum(pwr,dims=2));
        return pwr;
    else
        pwr = real.(X .* conj.(X));
        return pwr;
    end; 
end;

Base.length(pfbsch::FloatPFBScheme) = 1;
Base.iterate(pfbsch::FloatPFBScheme) = (pfbsch, nothing);
Base.iterate(pfbsch::FloatPFBScheme, state::Nothing) = nothing;

"""
```
RunPFB(pfbsch::FloatPFBScheme; data::Array{<:Complex}=Nothing) 
    :: Union{Tuple{Array{<:Real},Array{<:Real}},Array{<:Real}}
```
Here one parses a data vector to the PFB to run. Note its length must be a multiple
of N if a data file was not specified before in the scheme.
"""
function RunPFB(pfbsch::FloatPFBScheme; data::Array{<:Complex}=Nothing) #:: Union{Tuple{Array{<:Real},Array{<:Real}},Array{<:Real}}
    if isnothing(data)
        #Here we would look to the input file TODO.
    else                                                        #if we are using an input data array
        inputdata = data;
    end;
    size = length(data);                                    #get length of data stream
    com_cycles = div(size,pfbsch.N);                        #how many cycles of commutator
    data_slc = (1-pfbsch.N):0;                              #start with negative slice to make for loops simpler
    if pfbsch.staged                                            #if storing staged data
        X = Array{ComplexF64}(undef,(pfbsch.N,com_cycles,Int(log2(pfbsch.N)+2)));
                                                                #will be tapsize x datalen/point x stages
        for i in 1:com_cycles                                   #for each stage, populate all firs, and run FFT once
            X[:,i,:] .= natInIterDitFFT(pfbsch,PFBFir(pfbsch,inputdata[data_slc.+(pfbsch.N*(i))]));
        end;
    else                                                        #if storing staged data
        X = Array{ComplexF64}(undef,(pfbsch.N,com_cycles));     #will be tapsize x stages
        for j in 1:com_cycles                                   #for each stage, populate all firs, and run FFT once
            X[:,j] .= natInIterDitFFT(pfbsch,PFBFir(pfbsch,inputdata[data_slc.+(pfbsch.N*(j))]));
        end;
    end;
    if pfbsch.dual & ~pfbsch.staged                             #If dual processing but not staged                      
        # return Tuple(SpecPow.(pfbsch, SpecSplit(X)));
        return SpecSplit(X);
    elseif ~pfbsch.dual & pfbsch.staged                         #If single pol processing and staged
        return SpecPow(pfbsch, X[:,:,end]);
    elseif pfbsch.dual & pfbsch.staged                          #If dual pol and staged                
        return Tuple(SpecPow.(pfbsch, SpecSplit(X[:,:,end])));
    else                                                        #If single pol and no staging
        return SpecPow(pfbsch, X);
    end;
end;