using Test
using Base: abs
include("fixpoint.jl");
f_scheme = FixpointScheme(18,17);
v1 = [0.15566998822, 0.0001, 0.45];
v2 = [0.333222111991, 0.3344888, 0.779]

val1 = [0.15566998822 0.0001 0.45; 0.119 0.55 0.37711];
val2 = [0.333222111991 0.3344888 0.779; 0.123 0.6622 0.4621];



f_val1 = fromFloat(v1,f_scheme);
f_val2 = fromFloat(v2,f_scheme);

ideal_sum = v1 .+ v2;

f_sum = f_val1 + f_val2;
println(toFloat(f_sum))
println(ideal_sum)
error_sum =  abs.(toFloat(f_sum).-ideal_sum)
println(error_sum)

@test any(abs.(toFloat(f_sum)-ideal_sum) .< 0.001)
