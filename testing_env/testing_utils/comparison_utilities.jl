# Import necessary packages
using JSON3
using LinearAlgebra
using Setfield
# Define the function
function load_and_set_weights_MADE_relu_conditional(ps, json_file_path::String)
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

    @set! ps.internal_layer_1.layer_1.weight = block_0_context_layer_weights_matrix
    @set! ps.internal_layer_1.layer_1.bias = block_0_context_layer_bias_matrix

    @set! ps.internal_layer_1.layer_2.weight = linear_layer_0_block_0_weights_matrix
    @set! ps.internal_layer_1.layer_2.bias = linear_layer_0_block_0_bias_matrix

    @set! ps.internal_layer_1.layer_3.weight = linear_layer_1_block_0_weights_matrix
    @set! ps.internal_layer_1.layer_3.bias = linear_layer_1_block_0_bias_matrix

    @set! ps.internal_layer_2.layer_1.weight = block_1_context_layer_weights_matrix
    @set! ps.internal_layer_2.layer_1.bias = block_1_context_layer_bias_matrix

    @set! ps.internal_layer_2.layer_2.weight = linear_layer_0_block_1_weights_matrix
    @set! ps.internal_layer_2.layer_2.bias = linear_layer_0_block_1_bias_matrix

    @set! ps.internal_layer_2.layer_3.weight = linear_layer_1_block_1_weights_matrix
    @set! ps.internal_layer_2.layer_3.bias = linear_layer_1_block_1_bias_matrix

    @set! ps.final_layer.weight = final_layer_weights_matrix
    @set! ps.final_layer.bias = final_layer_bias_matrix

    return ps
end


function load_and_set_weights_context_encoder(ps, json_file_path::String)
    # Read and parse the JSON file
    layer_parameters = JSON3.read(open(json_file_path))

    # Access the imported parameters
    context_encoder_weights = layer_parameters[:context_encoder][:weights]
    context_encoder_weights_matrix = Matrix(hcat(Vector{Float32}.(context_encoder_weights)...)')
    #context_encoder_weights_matrix[[1,2,3,4], :] = context_encoder_weights_matrix[[2,4,1,3], :]

    context_encoder_bias = layer_parameters[:context_encoder][:bias]
    context_encoder_bias_matrix = Vector{Float32}(context_encoder_bias)
    #context_encoder_bias_matrix[[1,2,3,4], :] = context_encoder_bias_matrix[[2,4,1,3], :]

    # Set the weights and biases in the Lux model
    @set! ps.weight = context_encoder_weights_matrix
    @set! ps.bias = context_encoder_bias_matrix

    return ps
end