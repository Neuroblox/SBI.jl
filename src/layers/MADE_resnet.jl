using Random
using Lux
using ConcreteStructs
using Static
using SIMDTypes

const BoolType = Union{StaticBool, Bool, Val{true}, Val{false}}

include("../utils.jl")
include("Masked_layer.jl")


# MADE_resnet

#Currently we have linear layers and not masked linear layers involved in the resnet
# Next step is to add the masked linear layers

#and test them, might be worth throwing in a debug mode in the forward pass that outputs the output of each layer


@concrete struct MADE_relu <: Lux.AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    mask::Base.RefValue{}
    order::AbstractArray{Int}
end

function MADE_relu(in_dim, hidden_dim; gaussianMADE::Bool=true, random_order::Bool=false, order_permutation::Int=1)

    internal_layer = SkipConnection(Chain(MaskedLinear(hidden_dim,hidden_dim, relu), MaskedLinear(hidden_dim,hidden_dim, relu)),+)
    initial_layer = MaskedLinear(in_dim, hidden_dim, relu)
    final_layer = MaskedLinear(hidden_dim, in_dim*2)

    layers = NamedTuple{(:initial_layer, :internal_layer, :final_layer)}((initial_layer, internal_layer, final_layer)) 

    expanded_layers = layers.initial_layer, layers.internal_layer.layers[1], layers.internal_layer.layers[2], layers.final_layer

    m_k = generate_m_k(expanded_layers, random_order, order_permutation = order_permutation) # look uo exactly what scale random order means
    m_k[end-1] = m_k[2] # Check this doesnt mess with anything to bad needed to preserve masked properties

    order = m_k[1]

    mask = generate_masks(m_k, true)
    mask_ref = Ref(mask)

    for i in eachindex(expanded_layers)
        set_mask(expanded_layers[i],mask[i])
    end

    return MADE_relu(layers, mask_ref, order)
  
end
# -------------------------------------------------------------------
# MADE Container Layer


  #TODO Figure out why these generated funtions are used, probably for optimization reasons
  # essentially just the forward pass
  @generated function applyMADE_relu(layers::NamedTuple{fields}, x, ps,
    st::NamedTuple{fields}) where {fields}
  N = length(fields)
  x_symbols = vcat([:x], [gensym() for _ in 1:N])
  st_symbols = [gensym() for _ in 1:N]
  
  calls = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
    $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]
  
  push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
  #Add a debug checking
  push!(calls, :(return $(x_symbols[N + 1]), st))
  return Expr(:block, calls...)
  end


  
  
(c::MADE_relu)(x, ps, st::NamedTuple) = applyMADE_relu(c.layers, x, ps, st)  

function sample(T::MADE_relu, ps, st; samples = randn(T.layers[1].in_dims), use_softplus::Bool=false, debug = false)
  input = T.layers[1].in_dims
  order = sortperm(T.order) # gets the index for the m_k values in increasing order
  #println(samples)
  for i in order
    if use_softplus
      mean = T(samples, ps, st)[1][i]
      std = softplus.(T(samples, ps, st)[1][i+input])
    else
      mean = T(samples, ps, st)[1][i]
      std = exp(T(samples, ps, st)[1][i+input])
    end
    samples[i] = std*samples[i] + mean
    debug && println("Layer $i: mean = ", mean, ", std = ", std)
    debug && println(samples)
  end
  return samples
end
  