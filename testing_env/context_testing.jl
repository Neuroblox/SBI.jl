
# Note for future in MADE.py in nflows package what they do to residual blocks is
# run a single linear layer to go from the base value to the dimensions and run an activation function
# on this value. Then things proceed as normal. This shouldn't be too hard to implement, But it looks like we have to go back to 
# Square one and first implement a resnet conditional made, and then expand to the full maf. This should take a day to validate each, so 2 dayts worth of work.
# SHould be able to get it done by the end of the week. After this we can finally publish the package!!!!

#Same as the previous version except now I can go and explore the loss function
# keeping the previous version for documentation and validation reasons
using Pkg
Pkg.activate("./testing_env")
using Revise

using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi
using CairoMakie


import MLUtils: DataLoader, splitobs
include("../src/utils.jl")


rng = MersenneTwister()
Random.seed!(rng, 12345)
model = Sbi.MADE_relu_conditional(2, 4, 1);
#context = Sbi.context(1,4,1)

#=
function initialstates2(rng::AbstractRNG, l::AbstractLuxWrapperLayer{layer}) where {layer}
    println(getfield(l, layer))
    return Lux.initialstates(rng, Lux.getfield(l, layer))
end
=#

#=
function initialstates2(
    rng::AbstractRNG, l::Sbi.MADE_relu_conditional{layers}
) where {layers}
    return NamedTuple{layers}(Lux.initialstates.(rng, getfield.((l,), layers)))
end
=#



ps, state = Lux.setup(rng, model);

model(randn(3), ps, state)

#now I need to load in the weights and biases
using JSON3
using Setfield

# Filepath to the JSON file containing weights and biases
json_file_path = "/home/simon/Code/Sbi_JuliaVsPython/layer_parameters.json"

# Read and parse the JSON file
layer_parameters = JSON3.read(open(json_file_path))

# Access the imported parameters
initial_layer_weights = layer_parameters[:initial_layer][:weights]
initial_layer_weights_matrix = Matrix(hcat(Vector{Float32}.(initial_layer_weights)...)')

initial_layer_bias = layer_parameters[:initial_layer][:bias]
initial_layer_bias_matrix = Vector{Float32}(initial_layer_bias)

context_layer_weights = layer_parameters[:context_layer][:weights]
context_layer_weights_matrix = Matrix(hcat(Vector{Float32}.(context_layer_weights)...)')

context_layer_bias = layer_parameters[:context_layer][:bias]
context_layer_bias_matrix = Vector{Float32}(context_layer_bias)

linear_layer_0_block_0_weights = layer_parameters[:linear_layer_0_block_0][:weights]
linear_layer_0_block_0_weights_matrix = Matrix(hcat(Vector{Float32}.(linear_layer_0_block_0_weights)...)')

linear_layer_0_block_0_bias = layer_parameters[:linear_layer_0_block_0][:bias]
linear_layer_0_block_0_bias_matrix = Vector{Float32}(linear_layer_0_block_0_bias)

linear_layer_1_block_0_weights = layer_parameters[:linear_layer_1_block_0][:weights]
linear_layer_1_block_0_weights_matrix = Matrix(hcat(Vector{Float32}.(linear_layer_1_block_0_weights)...)')

linear_layer_1_block_0_bias = layer_parameters[:linear_layer_1_block_0][:bias]
linear_layer_1_block_0_bias_matrix = Vector{Float32}(linear_layer_1_block_0_bias)

block_0_context_layer_weights = layer_parameters[:block_0_context_layer][:weights]
block_0_context_layer_weights_matrix = Matrix(hcat(Vector{Float32}.(block_0_context_layer_weights)...)')

block_0_context_layer_bias = layer_parameters[:block_0_context_layer][:bias]
block_0_context_layer_bias_matrix = Vector{Float32}(block_0_context_layer_bias)

linear_layer_0_block_1_weights = layer_parameters[:linear_layer_0_block_1][:weights]
linear_layer_0_block_1_weights_matrix = Matrix(hcat(Vector{Float32}.(linear_layer_0_block_1_weights)...)')

linear_layer_0_block_1_bias = layer_parameters[:linear_layer_0_block_1][:bias]
linear_layer_0_block_1_bias_matrix = Vector{Float32}(linear_layer_0_block_1_bias)

linear_layer_1_block_1_weights = layer_parameters[:linear_layer_1_block_1][:weights]
linear_layer_1_block_1_weights_matrix = Matrix(hcat(Vector{Float32}.(linear_layer_1_block_1_weights)...)')

linear_layer_1_block_1_bias = layer_parameters[:linear_layer_1_block_1][:bias]
linear_layer_1_block_1_bias_matrix = Vector{Float32}(linear_layer_1_block_1_bias)

block_1_context_layer_weights = layer_parameters[:block_1_context_layer][:weights]
block_1_context_layer_weights_matrix = Matrix(hcat(Vector{Float32}.(block_1_context_layer_weights)...)')

block_1_context_layer_bias = layer_parameters[:block_1_context_layer][:bias]
block_1_context_layer_bias_matrix = Vector{Float32}(block_1_context_layer_bias)

final_layer_weights = layer_parameters["final_layer"]["weights"]
final_layer_weights_matrix = Matrix(hcat(Vector{Float32}.(final_layer_weights)...)')
final_layer_weights_matrix[[1,2,3,4], :] = final_layer_weights_matrix[[2,4,1,3], :]

final_layer_bias = layer_parameters["final_layer"]["bias"]
final_layer_bias_matrix = Vector{Float32}(final_layer_bias)
final_layer_bias_matrix[[1,2,3,4]] = final_layer_bias_matrix[[2,4,1,3]]

# Set the weights and biases in the Lux model
@set! ps.initial_layer.weight = initial_layer_weights_matrix
@set! ps.initial_layer.bias = initial_layer_bias_matrix

@set! ps.context_layer.weight = context_layer_weights_matrix
@set! ps.context_layer.bias = context_layer_bias_matrix



@set! ps.internal_layer.layer_1.weight = block_0_context_layer_weights_matrix
@set! ps.internal_layer.layer_1.bias = block_0_context_layer_bias_matrix

@set! ps.internal_layer.layer_2.weight = linear_layer_0_block_0_weights_matrix
@set! ps.internal_layer.layer_2.bias = linear_layer_0_block_0_bias_matrix

@set! ps.internal_layer.layer_3.weight = linear_layer_1_block_0_weights_matrix
@set! ps.internal_layer.layer_3.bias = linear_layer_1_block_0_bias_matrix


@set! ps.final_layer.weight = final_layer_weights_matrix
@set! ps.final_layer.bias = final_layer_bias_matrix

input = [-0.7943,  1.0887]
context_value = [1.0]
full_input = vcat(input, context_value)

final_output = model(full_input, ps, state)

#These are wrong write now (everything before this has been validated yaaaaayyyyyyyy!!)

@set! ps.internal_layer.layer_4.weight = block_1_context_layer_weights_matrix
@set! ps.internal_layer.layer_4.bias = block_1_context_layer_bias_matrix

@set! ps.internal_layer.layer_5.weight = linear_layer_0_block_1_weights_matrix
@set! ps.internal_layer.layer_5.bias = linear_layer_0_block_1_bias_matrix

@set! ps.internal_layer.layer_6.weight = linear_layer_1_block_1_weights_matrix
@set! ps.internal_layer.layer_6.bias = linear_layer_1_block_1_bias_matrix

@set! ps.final_layer.weight = final_layer_weights_matrix
@set! ps.final_layer.bias = final_layer_bias_matrix