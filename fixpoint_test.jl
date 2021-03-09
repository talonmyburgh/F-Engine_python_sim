using Test
using Base: abs
include("fixpoint.jl");

#Fixpoint testing
f_scheme = FixpointScheme(18,17);
v1 = [0.15566998822, 0.0001, 0.45];
v2 = [0.333222111991, 0.3344888, 0.779]

val1 = [0.15566998822 0.0001 0.45; 0.119 0.55 0.37711];
val2 = [0.333222111991 0.3344888 0.779; 0.123 0.6622 0.4621];

f_val1 = fromFloat(val1,f_scheme);
f_val2 = fromFloat(val2,f_scheme);

ideal_add = val1 .+ val2;
f_add = f_val1 + f_val2;

ideal_mul = val1 .* val2;
f_mul = f_val1 * f_val2;

ideal_sum=sum(val1,dims=1)  
f_sum = sum(f_val1,dims=1)

ideal_sub = val1 .- val2;
f_sub = f_val1 - f_val2;

@test any(abs.(toFloat(f_add)-ideal_add) .< 0.0001)
@test any(abs.(toFloat(f_mul)-ideal_mul) .< 0.0001)
@test any(abs.(toFloat(f_sum)-ideal_sum) .< 0.0001)
@test any(abs.(toFloat(f_sub)-ideal_sub) .< 0.0001)
@test any(abs.(toFloat(quantise(f_val1,f_scheme))-val1) .< 0.0001)

#CFixpoint testing
cf_val1 = fromComplex(val1,val2,f_scheme);
cf_val2 = fromComplex(val2, val1, f_scheme);

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

@test any(toComplex(cf_val1) != c_val1);
@test any(abs.(toComplex(c_add) .- ideal_cadd).< 0.0001)
@test any(abs.(toComplex(c_mul) .- ideal_cmul) .< 0.0001)
@test any(abs.(toComplex(c_min) .- ideal_cmin) .<0.0001)
@test any(abs.(toComplex(c_sum) .- ideal_csum) .<0.0001)
