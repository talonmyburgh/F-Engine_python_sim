using Test
using Base: abs
using Random
include("fixpoint.jl");

#Fixpoint testing. Create our scheme:
f_scheme_1 = FixpointScheme(18,17);
f_scheme_2 = FixpointScheme(12,13);

#Create our floating point vectors:
v1 = [0.15566998822, 0.0001, 0.45];
v2 = [0.333222111991, 0.3344888, 0.779]
val1 = [0.15566998822 0.0001 0.45; 0.119 0.55 0.37711];
val2 = [0.333222111991 0.3344888 0.779; 0.123 0.6622 0.4621];

#Cast them to fixed point using the above scheme
f_val1 = fromFloat(val1,f_scheme_1);
f_val2 = fromFloat(val2,f_scheme_1);

#Test adding
ideal_add = val1 .+ val2;
f_add = f_val1 + f_val2;

#Test multiplication
ideal_mul = val1 .* val2;
f_mul = f_val1 * f_val2;

#Test summing the vectors
ideal_sum=sum(val1,dims=1)  
f_sum = sum(f_val1,dims=1)

#Test subtracting the vectors
ideal_sub = val1 .- val2;
f_sub = f_val1 - f_val2;

#Test right shifting the vector values
ideal_rshift = f_val1.data .>> 1;
f_rshift = f_val1 >> 1;

#Test left shifting the vector values
ideal_lshift = f_val1.data .<< 1;
f_lshift = f_val1 << 1;

#Test indexing
f_val1[Array{Int, 2}([2 1 3; 3 1 2])]

@test any(abs.(toFloat(f_add)-ideal_add) .< 0.0001)
@test any(abs.(toFloat(f_mul)-ideal_mul) .< 0.0001)
@test any(abs.(toFloat(f_sum)-ideal_sum) .< 0.0001)
@test any(abs.(toFloat(f_sub)-ideal_sub) .< 0.0001)
@test any(abs.(f_rshift.data-ideal_rshift) .< 0.0001)
@test any(abs.(f_lshift.data-ideal_lshift) .< 0.0001)
@test any(abs.(toFloat(quantise(f_val1,f_scheme_2))-val1) .< 0.0001)

#CFixpoint testing
cf_val1 = fromComplex(val1,val2, f_scheme_1);
cf_val2 = fromComplex(val2, val1, f_scheme_1);

c_val1 = val1 + val2*im;
c_val2 = val2 + val1*im;

c_add = cf_val1 + cf_val2;
ideal_cadd = c_val1 .+ c_val2;

c_mul = cf_val1 * cf_val2;
ideal_cmul = c_val1 .* c_val2;

c_min = cf_val1 - cf_val2;
ideal_cmin = c_val1 .- c_val2;

c_sum = sum(cf_val1,dims=1);
ideal_csum = sum(c_val1,dims=1);

c_rshift = cf_val1 >> 1;
ideal_rshift_re = cf_val1.real.data .>> 1
ideal_rshift_im = cf_val1.imag.data .>> 1

@test any(toComplex(cf_val1) != c_val1);
@test any(abs.(toComplex(c_add) .- ideal_cadd).< 0.0001)
@test any(abs.(toComplex(c_mul) .- ideal_cmul) .< 0.0001)
@test any(abs.(toComplex(c_min) .- ideal_cmin) .<0.0001)
@test any(abs.(toComplex(c_sum) .- ideal_csum) .<0.0001)
@test any((abs.(c_rshift.real.data .- ideal_rshift_re).-abs.(c_rshift.imag.data .- ideal_rshift_im)) .<0.0001)
