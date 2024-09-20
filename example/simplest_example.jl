using Turing
using MLUtils
using Distributions
using LinearAlgebra, Zygote, ADTypes, Optimisers, Lux, Random, Optimisers, CUDA

function generate_simple_test(θ)
     return θ .+ 1.1*rand(3)
end

prior = Product(Uniform.([-2,-2,-2], 2))

#Generate a dataloader from these

#= Generate DataLoader Function

     ----- input --------

     - BlackBox- The simulator, just a function that accepts an array of priors and outputs the data/sumary statistics
     - prior - A samplable Distribution representing the Bayesian prior
     - batchsize, the batchsize for the DataLoader
     - output_dimension: the combined dimension of the prior and summary statistics,
     - output_size: The total number of simulations in the DataLoader

     ----- output -------

     Dataloader, output format, batches of size batchsize with [priors;simulator outputs]
=#
function generate_dataloader(BlackBox, prior::Distributions.Sampleable, batchsize, output_dimension, output_size)
     # generate the undefined array
     data_complete = Array{Float64}(undef, output_dimension, output_size)

     #multithreaded for loop
     Threads.@threads for i in 1:output_size
          prior_values_x = rand(prior)
          data_complete[:,i] = vcat(prior_values_x,BlackBox(prior_values_x))
     end

     loader = DataLoader(data_complete; batchsize=batchsize, shuffle=true)
     return loader

end

b = generate_dataloader(generate_simple_test, prior, 10, 6, 100)


## Define the Neural Network Architecture

using Sbi

rng = MersenneTwister()
Random.seed!(rng, 12345)
opt = Adam(0.060)


MADE1 = conditional_MADE(MaskedLinear(6,4,relu), MaskedLinear(4,6))
MADE2 = conditional_MADE(MaskedLinear(6,4,relu), MaskedLinear(4,6), random_order=true)

# temporarily deactivate for debugging purposes
#model = conditional_MAF(MADE1,MADE2)

model = conditional_MADE(MaskedLinear(6,4,relu), MaskedLinear(4,6))

ps, st = Lux.setup(rng, model) 

tstate = Lux.Training.TrainState(model, ps, st, opt)

train_dataloader = b

vjp_rule = Lux.Training.AutoZygote()
ADTypes.AutoZygote()

function main(tstate::Lux.Training.TrainState, vjp, data_loader, epochs)
     for epochs in 1:epochs
          for data in data_loader
               grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,lux_gaussian_maf_loss, data, tstate)
               println("Epoch: $(epoch) || Loss: $(loss)")
               tstate = Lux.Training.apply_gradients(tstate, grads)
          end
     end
end

main(tstate, vjp_rule, train_dataloader, 1000)