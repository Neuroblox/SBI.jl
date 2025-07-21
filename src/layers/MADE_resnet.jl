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
    initial_layer = MaskedLinear(in_dim, hidden_dim)
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
    #samples[i] = std*samples[i] + mean
    samples[i] = (samples[i]-mean)./std
    debug && println("Layer $i: mean = ", mean, ", std = ", std)
    debug && println(samples)
  end
  return samples
end


  
@concrete struct MADE_relu_conditional <: Lux.AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    mask::Base.RefValue{}
    order::AbstractArray{Int}
    context_dim::Int
end

function MADE_relu_conditional(in_dim, hidden_dim, context_dim; gaussianMADE::Bool=true, random_order::Bool=false, order_permutation::Int=1, internal_layer_num::Int=1)

    internal_layers = [SkipConnection(Chain(context(context_dim, hidden_dim, relu), MaskedLinear(hidden_dim,hidden_dim, relu), MaskedLinear(hidden_dim,hidden_dim, relu)),+) for _ in 1:internal_layer_num] 
    initial_layer = MaskedLinear(in_dim, hidden_dim)
    context_layer = context(context_dim, hidden_dim, relu)
    final_layer = MaskedLinear(hidden_dim, in_dim*2)

    #create symbol names for internal layers
    internal_layer_symbols = Tuple(Symbol("internal_layer_$(i)") for i in 1:internal_layer_num)

    layers = NamedTuple{(:initial_layer, :context_layer, internal_layer_symbols..., :final_layer)}((initial_layer, context_layer, internal_layers..., final_layer)) 

    # Double check the logic behind this (internal_layer[1] is a context layer so I dont think the mask should be set like this)
    expanded_layers = layers.initial_layer, [layers[i+2].layers[j] for i in 1:internal_layer_num, j in 2:3]..., layers.final_layer

    m_k = generate_m_k(expanded_layers, random_order, order_permutation = order_permutation) # look uo exactly what scale random order means
    m_k[end-1] = m_k[2] # Check this doesnt mess with anything to bad needed to preserve masked properties

    order = m_k[1]

    mask = generate_masks(m_k, true)
    mask_ref = Ref(mask)

    for i in eachindex(expanded_layers)
        set_mask(expanded_layers[i],mask[i])
    end

    return MADE_relu_conditional(layers, mask_ref, order, context_dim)
  
end



function Lux.initialstates(rng::AbstractRNG, l::MADE_relu_conditional{layers}) where {layers}
  print("usining MADE relu initial states")
  ctx = (context = l.layers.context_layer.in_dims,)
  other = invoke(Lux.initialstates, Tuple{AbstractRNG, Lux.AbstractLuxWrapperLayer}, rng, l)
  #other = context_state_finder(other, ctx.context)
  #standard = NamedTuple{layers}(Lux.initialstates.(rng, getfield.((l,), layers)))
  
  println("hi", ctx)
  println(other)
  return merge(ctx, other)
end



  


#This function iterates through the symbols in a named Tuple
# If the symbol is called context_layer, we modify the state context in context layer
function context_state_finder(st::NamedTuple, context_val)
  for k in keys(st)
    if k == :context_layer
      st = merge(st,(context_layer = (context = context_val,),))
    elseif startswith(string(k), "internal_layer")
      println("found internal layer", k)
      internal_st = st[k]
      println("internal_st", internal_st)
      internal_st = merge(internal_st, (layer_1 = (context = context_val,),)) # Assumes layer 1 is a context layer

      st = merge(st, NamedTuple{(k,)}((internal_st,)))
    end
  end
  return st
end

# quick test for context_state_finder
st = (context_layer = NamedTuple(), other_layer = (other = 1.2,))

#This function will be called in the apply
#Gets the context dimension from the state variable
#seperates the context from the Input
#sets the context state variable for the sub layers
#returns the modified state and input
function set_context(st::NamedTuple, x::AbstractVecOrMat)
  context_dim = st.context
  context = x[end-context_dim+1:end]
  x = x[1:end-context_dim]
  st = context_state_finder(st, context)
  println("context", context)
  println("x", x)
  return st, x
end


function debug_merge(st1, st2)
  # Merge the two states and print the debug information
  println("Debugging merge of states:")
  println("State 1:", st1)
  println("State 2:", st2)
  merged_st = merge(st1, st2)
  println("Merged state:", merged_st)
  return merged_st
end

function (c::MADE_relu_conditional)(x, ps, st::NamedTuple)
  println("using custom dispatch, MADE_relu_conditional")
  st, x = set_context(st, x)
  println("st", st)
  return applyMADE_relu_conditional(c.layers, x, ps, st)
end

# -------------------------------------------------------------------
# MADE Container Layer


  #TODO Figure out why these generated funtions are used, probably for optimization reasons
  # essentially just the forward pass

  #how did the old conditionals work?
  # Note now we have the context thing to worry about
  #removed fields from st
  @generated function applyMADE_relu_conditional(layers::NamedTuple{fields}, x, ps,
    st::NamedTuple) where {fields}
  N = length(fields)
  x_symbols = vcat([:x], [gensym() for _ in 1:N])
  st_symbols = [gensym() for _ in 1:N]

  calls = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
    $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]
  
  start = [:($(:og_st) = st)]
  calls = vcat(start, calls)
  push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
  push!(calls, :(st = debug_merge(og_st, st))) # merge the debug state with the original state
  #Add a debug checking
  push!(calls, :(return $(x_symbols[N + 1]), st))
  return Expr(:block, calls...)
  end


# Context_layer to make interface simple

@concrete struct context <: Lux.AbstractLuxLayer
  activation
  in_dims::Int
  out_dims::Int
  init_weight
  init_bias
  use_bias <: StaticBool
end

function context(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform,
        init_bias=zeros32, use_bias::BoolType=True())
  return context(activation, in_dims, out_dims, init_weight, init_bias, use_bias)
end


function Lux.initialparameters(rng::AbstractRNG, d::context)
    return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
        bias=d.init_bias(rng, d.out_dims, 1))
end



function Lux.parameterlength(d::context)
    return d.out_dims * (d.in_dims + 1)
end

# good for efficiency not exactly sure why yet
Lux.statelength(d::context) = 0


# modified standard dense layer to implement the mask value pointed to by the pointer
@inline function (d::context)(x::AbstractVecOrMat, ps, st::NamedTuple)
    context = st.context
    println("context", context)
    println("ps", ps)
    println("context_output", x .+ d.activation.((ps.weight) * context .+ ps.bias))
    return x .+ d.activation.((ps.weight)*context .+ ps.bias), st
end

# fix any issues with the context lauyer mask, need to check theory here
function set_mask(layer::context, mask)
end



# create the wrapper for the MADE_relu_conditional layer
# just applies the forward transform
# Untested, lets test it lol
@concrete struct MADE_relu_conditional_transform <: Lux.AbstractLuxWrapperLayer{:layers}
    layers <: NamedTuple
    context_dims ::Int
end

# Constructor for the MADE_relu_conditional_transform layer
#input should be the MADE_relu_conditional layer and a context encoder layer
function MADE_relu_conditional_transform(layers...; context_dims=1)
  # check length of layers and make sure its length 2 or throw an expr_forward
  if length(layers) != 2
    throw(ArgumentError("MADE_relu_conditional_transform requires 2 layers"))
  end
  
  #check that the first element of layer is the right type
  if !(isa(layers[1], MADE_relu_conditional))
    throw(ArgumentError("MADE_relu_conditional_transform requires first layer to be MADE_relu_conditional"))
  end

  # create named tuple for the layers
  layers = NamedTuple{(:MADE_relu_conditional, :context_encoder)}((layers[1], layers[2]))


  return MADE_relu_conditional_transform(layers, context_dims)
end

# Define the forward mode behavior
function (c::MADE_relu_conditional_transform)(x, ps, st::NamedTuple)
    #=
    context_dims = c.context_dims
    context_val = x[end-context_dims+1:end]
    st = merge(st,(MADE_relu_conditional_transform = (context = context_val,),))
    =#
    return applyMADE_relu_conditional_transform(c.layers, x, ps, st)
end


# Run the coordinate transform on the MADE_relu_conditional layer


function applyMADE_relu_conditional_transform(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple) where {fields}
  context_dims = st.context_dims
  context = x[end-context_dims+1:end] # get the context from the input
  x_no_context = x[1:end-context_dims] # remove the context from the input
  encoder_output, st1 = Lux.apply(layers.context_encoder, context, ps.context_encoder, st.context_encoder)
  # need to create add a coord_transform function that takes the encoder output as parameters
  MADE_output, st2 = Lux.apply(layers.MADE_relu_conditional, x, ps.MADE_relu_conditional, st.MADE_relu_conditional)

  #create a named tuple with st1 and st2
  st3 = NamedTuple{fields}((st2, st1,))
  # Made_outout_state
  st_MADE = (MADE_output = MADE_output, encoder_output = encoder_output,)

  st = merge(st, st3)
  st = merge(st, st_MADE)
  # apply the coordinate transform
  # need to know at what state is the coordinate transform applies
  # lets define coordinate_transform function

  reg_output = forward(x_no_context, MADE_output)#normal output placeholder
  println("x_no_context", x_no_context)
  println("MADE_output", MADE_output)
  println("encoder_output", encoder_output)
  println("reg_output", reg_output)
  println("inverse_exp", inverse_exp(reg_output, encoder_output))


  return inverse_exp(reg_output, encoder_output), st
end

function Lux.initialstates(rng::AbstractRNG, l::MADE_relu_conditional_transform{layers}) where {layers}
  print("using MADE relu conditional transform initial states")
  ctx = (context_dims = l.context_dims,)
  other = invoke(Lux.initialstates, Tuple{AbstractRNG, Lux.AbstractLuxWrapperLayer}, rng, l)
  #other = context_state_finder(other, ctx.context)
  #standard = NamedTuple{layers}(Lux.initialstates.(rng, getfield.((l,), layers)))
  println("hi", ctx)
  println(other)
  return merge(ctx, other)
end