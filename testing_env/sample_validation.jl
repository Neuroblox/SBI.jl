
#Same as the previous version except now I can go and explore the loss function
# keeping the previous version for documentation and validation reasons
using Pkg
Pkg.activate("./testing_env")
using Revise

using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi
using CairoMakie


import MLUtils: DataLoader, splitobs
include("src/utils.jl")


rng = MersenneTwister()
Random.seed!(rng, 12345)
model1 = MADE_relu(2, 4)


simple_model = MAF(model1, softplus=true)

ps, st = Lux.setup(rng, simple_model)

#Next check weigh setting example should be in actual test scripts

ps.MADE_1.initial_layer.weight

#we know how to copy parameters. Now I need to mfigure out how to run a single forward in python

#Lets get a test x and output, have that be part of the test
#verify with hand calculations

# first lets copy the python implementation

#import the objet from layer_parameters.json
#using JSON3

#filepath: /home/simon/Code/SBI.jl/testing_env/layer_parameters.json


using JSON3

# Filepath to the JSON file
json_file_path = "testing_env/testing_utils/layer_parameters_made.json"

# Read and parse the JSON file
layer_parameters = JSON3.read(open(json_file_path))

# Access the imported parameters
println(layer_parameters["final_layer"]["weights"])

final_layer_weights = layer_parameters["final_layer"]["weights"]
final_layer_weights_matrix = Matrix(hcat(Vector{Float32}.(final_layer_weights)...)')
final_layer_weights_matrix[[1,2,3,4], :] = final_layer_weights_matrix[[2,4,1,3], :]

final_layer_bias = layer_parameters["final_layer"]["bias"]
final_layer_bias_matrix = Vector{Float32}(final_layer_bias)
final_layer_bias_matrix[[1,2,3,4]] = final_layer_bias_matrix[[2,4,1,3]]


initial_layer_weights = layer_parameters["initial_layer"]["weights"]
initial_layer_weights_matrix = Matrix(hcat(Vector{Float32}.(initial_layer_weights)...)')
# same thing but fore bias
initial_layer_bias = layer_parameters["initial_layer"]["bias"]
initial_layer_bias_matrix = Vector{Float32}(initial_layer_bias)

# internal layers
internal_layer_0_weights = layer_parameters["linear_layer_0"]["weights"]
internal_layer_0_weights_matrix = Matrix(hcat(Vector{Float32}.(internal_layer_0_weights)...)')

internal_layer_0_bias = layer_parameters["linear_layer_0"]["bias"]
internal_layer_0_bias_matrix = Vector{Float32}(internal_layer_0_bias)

# internal layer 1
internal_layer_1_weights = layer_parameters["linear_layer_1"]["weights"]
internal_layer_1_weights_matrix = Matrix(hcat(Vector{Float32}.(internal_layer_1_weights)...)')

internal_layer_1_bias = layer_parameters["linear_layer_1"]["bias"]
internal_layer_1_bias_matrix = Vector{Float32}(internal_layer_1_bias)

using Setfield
@set! ps.MADE_1.initial_layer.weight = initial_layer_weights_matrix
@set! ps.MADE_1.initial_layer.bias = initial_layer_bias_matrix

#internal layers
@set! ps.MADE_1.internal_layer.layer_1.weight = internal_layer_0_weights_matrix
@set! ps.MADE_1.internal_layer.layer_1.bias = internal_layer_0_bias_matrix
@set! ps.MADE_1.internal_layer.layer_2.weight = internal_layer_1_weights_matrix
@set! ps.MADE_1.internal_layer.layer_2.bias = internal_layer_1_bias_matrix

#final layer
@set! ps.MADE_1.final_layer.weight = final_layer_weights_matrix


#switch bias based of how we defined mask
@set! ps.MADE_1.final_layer.bias = final_layer_bias_matrix

model1.mask[][1]

#Here is the benchmark value
input = [-1.2111, -1.2802]
#reverse this
input2 = [-0.5067, -1.1271]

m_in =[-0.3943 1.0887 ;0.7718  0.3212; -1.5811 -0.3818]
#get the initial layer
initial_layer = model1.layers.initial_layer
middle_layer = model1.layers.internal_layer
final_layer = model1.layers.final_layer


#push benchmark value forward
output = initial_layer(m_in[:, [2,1]]', ps.MADE_1.initial_layer, st.MADE_1.initial_layer)

isapprox(output[1], [-1.0740  0.2025 -0.3205  0.5242;
                -0.7969  0.2154 -0.1492  0.3300;
                -0.5430  0.2273  0.0077  0.1521]', atol=1e-3)


output2 = middle_layer(output[1], ps.MADE_1.internal_layer, st.MADE_1.internal_layer)

output3 = final_layer(output[1], ps.MADE_1.final_layer, st.MADE_1.final_layer)

complete_output = simple_model(input, ps, st)

# Now lets get the loss function
# Note I also need to validate the transformation, but we can do that later

#get input data validated in python so we know what the loss is supposed to be
input = [ -0.61526, 0.9918]

lux_gaussian_maf_loss(simple_model, ps, st, input)

inverse(input[[1,2]], [0.16488593816757202, -0.2034382622581793, 0.31678903102874756, -0.3913199595114242])

lux_gaussian_maf_loss(simple_model, ps, st, m_in[:, [2,1]]')


# now lets check sampling is working
test_input = [0.9918, -0.6152]
output = Sbi.sample(simple_model, ps, st, specific_sample = test_input[[1,2]])
back_to_input = simple_model(output[[1,2]], ps, st)
back_to_input = forward(output[[1,2]], back_to_input[1])
println(back_to_input)

#Note its weird that output is reversed from what we expect, will that cause any issues?
#No its because I dont have that reverse permutation layer,
#That explains it. Alright sample seems to be working fine


# Now we can validate for a single made_relu
# get a loss function plot for each aswe;; as a visualization of 500 samples, get for each
#store both in some sort of notebook, or just use a powerpoint
# Actually markdown would be perfect for this sort of thing


