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
model = Sbi.MADE_relu_conditional(2, 4, 1, internal_layer_num=2);
context_encoder = Dense(1=>4, Lux.relu)
model_2 = Sbi.MADE_relu_conditional_transform(model, context_encoder, context_dims=1)
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



ps, state = Lux.setup(rng, model_2);

l, st = model_2(randn(3), ps, state)
