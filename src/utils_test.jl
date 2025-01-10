include("utils.jl")

#test for inverse function
made_output = [1 2; 3 4; log(5) log(6); log(7) log(8)]
u = [1 1; 3 4]

ground_truth = [(log(6) + 0.001 +1) (log(7) + 0.001 +2); (3*(log(8)+ 0.001)+3) (4*(log(9) + 0.001)+4)] 
inverse_transform = inverse(u, made_output)

isapprox(ground_truth, inverse_transform, atol=1e-5)

