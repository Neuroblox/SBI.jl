using Turing
using MLUtils
using Distributions

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




