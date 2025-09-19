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

    @set! ps.internal_layer_1.layer_3.weight = block_0_context_layer_weights_matrix
    @set! ps.internal_layer_1.layer_3.bias = block_0_context_layer_bias_matrix

    @set! ps.internal_layer_1.layer_2.weight = linear_layer_0_block_0_weights_matrix
    @set! ps.internal_layer_1.layer_2.bias = linear_layer_0_block_0_bias_matrix

    @set! ps.internal_layer_1.layer_5.weight = linear_layer_1_block_0_weights_matrix
    @set! ps.internal_layer_1.layer_5.bias = linear_layer_1_block_0_bias_matrix

    @set! ps.internal_layer_2.layer_3.weight = block_1_context_layer_weights_matrix
    @set! ps.internal_layer_2.layer_3.bias = block_1_context_layer_bias_matrix

    @set! ps.internal_layer_2.layer_2.weight = linear_layer_0_block_1_weights_matrix
    @set! ps.internal_layer_2.layer_2.bias = linear_layer_0_block_1_bias_matrix

    @set! ps.internal_layer_2.layer_5.weight = linear_layer_1_block_1_weights_matrix
    @set! ps.internal_layer_2.layer_5.bias = linear_layer_1_block_1_bias_matrix

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


function load_and_set_weights_MAF_relu_conditional(ps, json_file_path_1::String, json_file_path_2::String)
    # Read and parse the JSON file
    ps1 = load_and_set_weights_MADE_relu_conditional_transform(ps.MADE_1, json_file_path_1)
    ps2 = load_and_set_weights_MADE_relu_conditional_transform(ps.MADE_2, json_file_path_2)
    # reverse the inputs so we can stay consistent with the python benchmark validation
    @set! ps2.MADE_relu_conditional.initial_layer.weight = ps2.MADE_relu_conditional.initial_layer.weight[:, [2,1]] 
    ps = merge(ps, (MADE_1 = ps1, MADE_2 = ps2))
    return ps
end

function load_and_set_weights_MADE_relu_conditional_transform(ps, json_file_path_1::String)
    # Read and parse the JSON file
    ps_sub = load_and_set_weights_MADE_relu_conditional(ps.MADE_relu_conditional, json_file_path_1)

    ps_context = load_and_set_weights_context_encoder(ps.context_encoder, "./testing_env/testing_utils/context_encoder_parameters.json")
    ps = merge(ps, (MADE_relu_conditional = ps_sub, context_encoder = ps_context))
    return ps
end