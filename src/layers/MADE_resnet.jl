using Random
using Lux
using ConcreteStructs
using Static
using SIMDTypes

const BoolType = Union{StaticBool, Bool, Val{true}, Val{false}}

include("../utils.jl")


# MADE_resnet

#Currently we have linear layers and not masked linear layers involved in the resnet
# Next step is to add the masked linear layers

#and test them, might be worth throwing in a debug mode in the forward pass that outputs the output of each layer


@concrete struct MADE_relu <: Lux.AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
end

function MADE_relu(in_dim, hidden_dim; gaussianMADE::Bool=true, random_order::Bool=false)

    internal_layer = SkipConnection(Chain(Dense(hidden_dim,hidden_dim, relu), Dense(hidden_dim,hidden_dim, relu)),+)
    initial_layer = Dense(in_dim, hidden_dim, relu)
    final_layer = Dense(hidden_dim, in_dim)

    layers = NamedTuple{(:initial_layer, :internal_layer, :final_layer)}((initial_layer, internal_layer, final_layer)) 
  
  
    return MADE_relu(layers)
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
  push!(calls, :(return $(x_symbols[N + 1]), st))
  return Expr(:block, calls...)
  end


  
  
(c::MADE_relu)(x, ps, st::NamedTuple) = applyMADE_relu(c.layers, x, ps, st)  
  