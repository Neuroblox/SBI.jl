using Test
using Pkg
Pkg.activate("./testing_env")
using Revise

using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi
using CairoMakie

import MLUtils: DataLoader, splitobs
include("src/utils.jl")
include("testing_env/testing_utils/comparison_utilities.jl")

@testset "MADE_relu_conditional_transform Tests" begin
    
    # Setup
    rng = MersenneTwister()
    Random.seed!(rng, 12345)
    
    @testset "Basic MADE_relu_conditional Test" begin
        model = Sbi.MADE_relu_conditional(2, 4, 1, internal_layer_num=2)
        ps, state = Lux.setup(rng, model)
        
        ps = load_and_set_weights_MADE_relu_conditional(ps, "/home/simon/Code/SBI.jl/testing_env/testing_utils/layer_parameters.json")
        
        input = [1.0887, -0.3943]
        context_value = [1.0]
        full_input = vcat(input, context_value)
        
        final_output = model(full_input, ps, state)
        output = forward(full_input[1:2], final_output[1])
        
        @test isapprox(output[1], [1.1511, 0.2296], atol=1e-3)
        @test isapprox(output[2], -0.9639, atol=1e-2) # not exact, should investigate if I run into issues
    end
    
    @testset "MADE_relu_conditional_transform Single Input Test" begin
        model = Sbi.MADE_relu_conditional(2, 4, 1, internal_layer_num=2)
        ps, state = Lux.setup(rng, model)
        ps = load_and_set_weights_MADE_relu_conditional(ps, "/home/simon/Code/SBI.jl/testing_env/testing_utils/layer_parameters.json")
        
        context_encoder = Dense(1=>4)
        model_2 = Sbi.MADE_relu_conditional_transform(model, context_encoder, context_dims=1)
        
        ps2, state = Lux.setup(rng, model_2)
        context_ps, context_state = Lux.setup(rng, context_encoder)
        
        ps3 = load_and_set_weights_context_encoder(context_ps, "/home/simon/Code/SBI.jl/testing_env/testing_utils/context_encoder_parameters.json")
        ps2 = merge(ps2, (MADE_relu_conditional = ps, context_encoder = ps3))
        
        input = [1.0887, -0.3943]
        context_value = [1.0]
        full_input = vcat(input, context_value)
        
        l, st = model_2(full_input, ps2, state)
        logp = Sbi.logp_conditional_maf_smooth(l, st)
        
        @test isapprox(logp, -10.0594, atol=1e-2)
    end
    
    @testset "MADE_relu_conditional_transform Multiple Inputs Test" begin
        model = Sbi.MADE_relu_conditional(2, 4, 1, internal_layer_num=2)
        ps, state = Lux.setup(rng, model)
        ps = load_and_set_weights_MADE_relu_conditional(ps, "/home/simon/Code/SBI.jl/testing_env/testing_utils/layer_parameters.json")
        
        context_encoder = Dense(1=>4)
        model_2 = Sbi.MADE_relu_conditional_transform(model, context_encoder, context_dims=1)
        
        ps2, state = Lux.setup(rng, model_2)
        context_ps, context_state = Lux.setup(rng, context_encoder)
        
        ps3 = load_and_set_weights_context_encoder(context_ps, "/home/simon/Code/SBI.jl/testing_env/testing_utils/context_encoder_parameters.json")
        ps2 = merge(ps2, (MADE_relu_conditional = ps, context_encoder = ps3))
        
        input = [1.0887 -0.3943; 0.5 -0.5; 0.1 0.2]'
        context_value = [1.0 2.0 -1.0]
        full_input = vcat(input, context_value)
        
        l, st = model_2(full_input, ps2, state)
        
        rearranged = [0.1639   0.1639   0.1639;
                      0.3955  -0.1560  -0.3617;
                      0.3877   0.3877   0.3877;
                     -0.6512  -0.6731  -0.2559]
        
        @test isapprox(st.MADE_output, rearranged, atol=1e-3)
        
        logp = Sbi.logp_conditional_maf_smooth(l, st)
        @test isapprox(logp, -25.646, atol=1e-2)
    end
    
    @testset "ActivationLayer Test" begin
        model3 = Sbi.ActivationLayer(relu)
        ps3, state3 = Lux.setup(rng, model3)
        l2, st2 = model3([1.0, 2.3, -5.1], ps3, state3)
        
        # Test that ReLU activation works correctly
        expected = [1.0, 2.3, 0.0]  # ReLU should zero out negative values
        @test isapprox(l2, expected, atol=1e-6)
    end
end