
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
include("./testing_utils/comparison_utilities.jl")


rng = MersenneTwister()
Random.seed!(rng, 12345)
model = Sbi.MADE_relu_conditional(2, 4, 1, internal_layer_num=2);



ps, state = Lux.setup(rng, model);

ps = load_and_set_weights_MADE_relu_conditional(ps, "/home/simon/Code/SBI.jl/testing_env/testing_utils/layer_parameters.json")

#input = [-0.7943,  1.0887]
input = [1.0887, -0.3943]
context_value = [1.0]
full_input = vcat(input, context_value)

final_output = model(full_input, ps, state)

forward(full_input[1:2], final_output[1])

#Thing to test
isapprox(final_output[1], [-0.2093, 0.4175, -0.3117, 0.4304], atol=1e-2)


# now lets test the version with the transform
context_encoder = Dense(1=>4)

model_2 = Sbi.MADE_relu_conditional_transform(model, context_encoder, context_dims=1)

ps2, state = Lux.setup(rng, model_2);

context_ps ,context_state = Lux.setup(rng, context_encoder);

ps3 = load_and_set_weights_context_encoder(context_ps, "/home/simon/Code/SBI.jl/testing_env/testing_utils/context_encoder_parameters.json")

ps2 = merge(ps2, (MADE_relu_conditional = ps, context_encoder = ps3))

l, st = model_2(full_input, ps2, state)

isapprox(l, [ 3.7176, -0.8894], atol=1e-1)

logp = Sbi.logp_conditional_maf_smooth(l, st)
isapprox(logp, -8.4472, atol=1e-2)


#Now do for multiple inputs
input = [1.0887 -0.3943; 0.5 -0.5; 0.1 0.2]'
context_value = [1.0 2.0 -1.0]
full_input = vcat(input, context_value)


l, st = model_2(full_input, ps2, state)
isapprox(l, [ 0.9633 -0.5172;
                0.6723 -0.6783;
                0.4746 -0.0752]', atol=1e-1)