using Test
using Base: abs
include("fixpoint.jl");
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
