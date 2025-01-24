include("utils.jl")

#test for inverse function
made_output = [1 2; 3 4; log(5) log(6); log(7) log(8)]
u = [1 1; 3 4]

ground_truth = [(log(6) + 0.001 +1) (log(7) + 0.001 +2); (3*(log(8)+ 0.001)+3) (4*(log(9) + 0.001)+4)] 
inverse_transform = inverse(u, made_output)

isapprox(ground_truth, inverse_transform, atol=1e-5)

using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi
using CairoMakie

rng = MersenneTwister()
Random.seed!(rng, 12345)

# Set the optimizer model
opt = Adam(0.060)

model1 = MADE_relu(4, 20)
model2 = MADE_relu(4, 20, random_order=true)
model3 = MADE_relu(4, 20, random_order=true)

model = MAF(model1, model2, model3, softplus=true)

testx = rand(4,5)

ps, st = Lux.setup(rng, model)
tstate = Lux.Training.TrainState(model, ps, st, opt)

save_model(tstate, "test_model")

