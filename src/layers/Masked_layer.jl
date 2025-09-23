using Random
using Lux
using ConcreteStructs
using Static
using SIMDTypes

const BoolType = Union{StaticBool, Bool, Val{true}, Val{false}}

include("../utils.jl")


"""
MaskedLinear(in_dims => out_dims, activation=identity; init_weight=glorot_uniform,
          init_bias=zeros32, bias::Bool=true)

Added a traditional mask to a traditional fully connected layer, blocking certain connections. The forward pass is given by:
`y = activation.(weight * mask * x .+ bias)`

## Arguments

  - `in_dims`: number of input dimensions
  - `out_dims`: number of output dimensions
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`)
  - `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  - `allow_fast_activation`: If `true`, then certain activations can be approximated with
    a faster version. The new activation function will be given by
    `NNlib.fast_act(activation)`
  - 'init_mask': Initial mask, stored as a reference to allow for dynamic masks, default, all ones (no masking)

## Input

  - `x` must be an AbstractArray with `size(x, 1) == in_dims`

## Returns

- AbstractArray with dimensions `(out_dims, ...)` where `...` are the dimensions of `x`
- Empty `NamedTuple()`

## Parameters

- `weight`: Weight Matrix of size `(out_dims, in_dims)`
- `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)
"""
@concrete struct MaskedLinear <: Lux.AbstractLuxLayer
  activation
  in_dims::Int
  out_dims::Int
  init_weight
  init_bias
  use_bias <: StaticBool
  init_mask::Base.RefValue{Matrix{Float32}}
end

function Base.show(io::IO, d::MaskedLinear)
  print(io, "MaskedLinear($(d.in_dims) => $(d.out_dims)")
  return print(io, ")")
end

function MaskedLinear(mapping::Pair{<:Int, <:Int}; kwargs...)
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Masked_linear constructor called: kwargs=$kwargs")
  end
  return MaskedLinear(first(mapping), last(mapping); kwargs...)
end

# added a mak to the constructor
#used a reference since it needs to be mutable, and that cant happen with direct storage in a concrete structure
function MaskedLinear(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform,
        init_bias=zeros32, use_bias::BoolType=True())
        init_mask=ones(Float32, out_dims, in_dims)
        init_mask_ref = Ref(init_mask)
  return MaskedLinear(activation, in_dims, out_dims, init_weight, init_bias, use_bias, init_mask_ref)
end

function Lux.initialparameters(rng::AbstractRNG, d::MaskedLinear)
    return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
        bias=d.init_bias(rng, d.out_dims, 1))
end

function Lux.parameterlength(d::MaskedLinear)
    return d.out_dims * (d.in_dims + 1)
end

# good for efficiency not exactly sure why yet
Lux.statelength(d::MaskedLinear) = 0


# modified standard dense layer to implement the mask value pointed to by the pointer
@inline function (d::MaskedLinear)(x::AbstractVecOrMat, ps, st::NamedTuple)
  #log the layer parameters and computations for debugging
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("MaskedLinear forward pass: mask_size=$(size(d.init_mask[])), weight_size=$(size(ps.weight)), input_size=$(size(x)), input=$x, weight=$(ps.weight), bias=$(ps.bias), mask=$(d.init_mask[])")
  end
  output = d.activation.(((d.init_mask[]).*ps.weight)*x .+ ps.bias)
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("MaskedLinear output: output=$output")
  end
  return output, st
end


# -------------------------------------------------------------------
# MADE Container Layer


# TODO GIve a more detailed comment on this layer onsistent with the others
# MADE container - containter of MaskedLinear Layers (implemented as a Guassian Made)
@concrete struct MADE <: Lux.AbstractLuxWrapperLayer{:layers}
  layers <: NamedTuple
  mask::Base.RefValue{}
  order::AbstractArray{Int}
end

# Implements the sample function to sample from the distribution represented by the MADE container
function sample(T::MADE, ps, st; samples = randn(T.layers[1].in_dims), use_softplus::Bool=false)
  input = T.layers[1].in_dims
  order = sortperm(T.order) # gets the index for the m_k values in increasing order
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("MADE sampling: initial_samples=$samples")
  end
  for i in order
    if use_softplus
      mean = T(samples, ps, st)[1][i]
      std = softplus.(T(samples, ps, st)[1][i+input])
    else
      mean = T(samples, ps, st)[1][i]
      std = exp(T(samples, ps, st)[1][i+input])
    end
    samples[i] = std*samples[i] + mean
  end
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("MADE sampling complete: final_samples=$samples")
  end
  return samples
end


#Generates a seet of integers for a layer consistent with the autoregressive property
#used to calculate the mask
function generate_m_k(layers, random_order::Bool; num_conditional=0, order_permutation = 1)
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Generating m_k: layers=$layers, num_conditional=$num_conditional, order_permutation=$order_permutation")
  end

  dims = [(i.in_dims, i.out_dims) for i in layers]
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Layer dimensions: dims=$dims")
  end

  D = dims[1][1]
  D = D - num_conditional

  integer_assign = []
  if random_order
    push!(integer_assign, randperm(D))
  else
    if order_permutation == 1
      push!(integer_assign, 1:D)
    else
      push!(integer_assign, D:-1:1)
    end
  end

  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Initial integer assignment: D=$D, integer_assign=$integer_assign")
  end

  for i in dims[1:end-1]
    push!(integer_assign, rand(1:D-1, i[2])) #TODO double check the integer assign is working
  end
  push!(integer_assign,integer_assign[1])

  integer_assign[1] = vcat(integer_assign[1], ones(Int, num_conditional))

  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Final integer assignment: integer_assign=$integer_assign")
  end
  return(integer_assign)
end


#Calculate masks for each layer and pushes them to an array in order to be sent to the layer
#gaussianMADE only one implemented 
function generate_masks(m_k, gaussianMADE::Bool)
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Generating masks: m_k=$m_k, gaussianMADE=$gaussianMADE")
  end
  Masks = []
  for i in eachindex(m_k[1:end-2]) 
    pair = collect(Iterators.product(m_k[i], m_k[i+1]))
    M = zeros(size(pair))
    foreach(i -> pair[i][1] > pair[i][2] ?  M[i] = 0 : M[i] = 1, CartesianIndices(pair))
    push!(Masks, M')
  end

  pair = collect(Iterators.product(m_k[end-1], m_k[end]))
  M = zeros(size(pair))
  foreach(i -> pair[i][1] >= pair[i][2] ?  M[i] = 0 : M[i] = 1, CartesianIndices(pair))

  if gaussianMADE
    M = hcat(M,M)
  end

  push!(Masks, M')

  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Generated masks: Masks=$Masks")
  end
  return(Masks)
end


#Constructor for the MADE layer
# sets initial mask
function MADE(layers...; gaussianMADE::Bool=true, random_order::Bool=false)
  names = ntuple(i -> Symbol("layer_$i"), length(layers))

  m_k = generate_m_k(layers, random_order)

  order = m_k[1]

  mask = generate_masks(m_k, gaussianMADE)
  mask_ref = Ref(mask)

  for i in eachindex(layers)
    set_mask(layers[i],mask[i])
  end

  return MADE(NamedTuple{names}(layers), mask_ref, order)
end

# Just Sets Mask for a particular layer
function set_mask(layer::MaskedLinear, mask)
  layer.init_mask[] = mask
end


(c::MADE)(x, ps, st::NamedTuple) = applyMADE(c.layers, x, ps, st, c.mask)

#TODO Figure out why these generated funtions are used, probably for optimization reasons
# essentially just the forward pass
@generated function applyMADE(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}, masks) where {fields}
N = length(fields)
x_symbols = vcat([:x], [gensym() for _ in 1:N])
st_symbols = [gensym() for _ in 1:N]

calls = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]

push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st))
return Expr(:block, calls...)
end


MADE(; kwargs...) = MADE((; kwargs...))



#-------------------------------------------------------------------------------------------------------------

# MADE conditional Container Layer


# TODO GIve a more detailed comment on this layer onsistent with the others
# MADE container - containter of MaskedLinear Layers (implemented as a Guassian Made)

struct conditional_MADE{T <: NamedTuple} <: Lux.AbstractLuxWrapperLayer{(:layers)}
  layers::T
  mask::Base.RefValue{}
  order::AbstractArray{Int}
  num_conditional::Int
end

function sample(T::conditional_MADE, ps, st; samples = randn(T.layers[1].in_dims))
  input = T.layers[1].in_dims
  output = T.layers[end].out_dims
  non_conditional_input = div(output,2)
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Conditional MADE sampling: input=$input, output=$output, non_conditional_input=$non_conditional_input")
  end
  input_m_k = copy(T.order[1:non_conditional_input])
  order = sortperm(input_m_k) # gets the index for the m_k values in increasing order
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Sampling order: order=$order, T_order=$(T.order)")
  end
  order = order[1:non_conditional_input,:]
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Initial samples: samples=$samples")
  end
  for i in order
    mean = T(samples, ps, st)[1][i]
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
      println("Sampling step: i=$i, non_conditional_input=$non_conditional_input")
    end
    std = exp(T(samples, ps, st)[1][i+non_conditional_input ])
    samples[i] = std*samples[i] + mean
  end
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Final samples: samples=$samples")
  end
  return samples
end


#Constructor for the MADE layer
# sets initial mask
function conditional_MADE(layers...; gaussianMADE::Bool=true, random_order::Bool=false)
  names = ntuple(i -> Symbol("layer_$i"), length(layers))

  input_size = layers[1].in_dims
  output_size = layers[end].out_dims

  num_conditional = Int(input_size - (output_size / 2))

  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Conditional MADE constructor: num_conditional=$num_conditional, layers=$layers")
  end

  m_k = generate_m_k(layers, random_order, num_conditional=num_conditional)

  order = m_k[1]

  mask = generate_masks(m_k, gaussianMADE)
  mask_ref = Ref(mask)

  for i in eachindex(layers)
    set_mask(layers[i],mask[i])
  end

  return conditional_MADE(NamedTuple{names}(layers), mask_ref, order, num_conditional)
end

(c::conditional_MADE)(x, ps, st::NamedTuple) = apply_conditionalMADE(c.layers, x, ps, st)

#TODO Figure out why these generated funtions are used, probably for optimization reasons
# essentially just the forward pass
@generated function apply_conditionalMADE(layers::NamedTuple{fields}, x, ps,
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


conditional_MADE(; kwargs...) = conditional_MADE((; kwargs...))

#-------------------------------------------------------------------------------------------------------------
# MAF layer (chain of MADE)


@concrete struct MAF <: Lux.AbstractLuxWrapperLayer{:layers}
  layers <: NamedTuple
  softplus::Bool
end

function MAF(layers...; softplus::Bool=false)
  names = ntuple(i -> Symbol("MADE_$i"), length(layers))
  return MAF(NamedTuple{names}(layers), softplus)
end

(c::MAF)(x, ps, st::NamedTuple) = c.softplus ? applyMAF_smooth(c.layers, x, ps, st) : applyMAF(c.layers, x, ps, st)
#(c::MAF)(x, ps, st::NamedTuple) = applyMAF(c.layers, x, ps, st)

# simple macro that transforms x to there correspoding random variable representation
#used in the flow part of Masked autoregressive flow
@inline function coord_transform(x, y_pred)
    n = div(size(y_pred)[1], 2)
    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("Coordinate transform: x_sample=$(x[:,1]), half1_sample=$(half1[:,1]), half2_sample=$(half2[:,1]), y_pred_sample=$(y_pred[:,1])")
    end
    u = (x .- half1).*exp.(-half2)
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("Coordinate transform result: u_sample=$(u[:,1])")
    end
  return u
end


# simple macro that transforms x to there correspoding random variable representation
#used in the flow part of Masked autoregressive flow
# Note smooth version should give better stability in training
@inline function coord_transform_smooth(x, y_pred)
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Smooth coordinate transform: x=$x, y_pred=$y_pred")
  end
  u = forward(x, y_pred)
  if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
    println("Smooth coordinate transform result: u=$u")
  end
return u
end

# forward pass, use the coord transform
# TODO Test this and make sure its not causing the bug that keeps coming up
@generated function applyMAF(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}) where {fields}
N = length(fields) #number of MADE layers
x_symbols = vcat([:x], [gensym() for _ in 1:N])
total_std = [gensym() for _ in 1:N]
st_symbols = [gensym() for _ in 1:N]
calls1 = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]
calls2 = [:($(x_symbols[i]) = coord_transform($(x_symbols[i-1]),$(x_symbols[i]))) for i in 2:N]
calls3 = [:($(total_std[i]) = copy($(x_symbols[i+1]))) for i in 1:N]



n = length(calls1) + length(calls2) + length(calls3)
calls = similar(calls1, n)

#add up all the log std for each layer
#each_layer_ouptut = :([$(x_symbols[N]) for i in 1:N])

################# add the definition of total_std as an array in the list of blocks
#calls[1] .= :($(x_symbols[1]) = 1)
calls[1:3:n] .= calls1
calls[2:3:n] .= calls3
calls[3:3:n] .= calls2


push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st, $(x_symbols[N]), $(total_std...)))
return Expr(:block, calls...)
end


# forward pass, use the coord transform
# TODO Test this and make sure its not causing the bug that keeps coming up
@generated function applyMAF_smooth(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}) where {fields}
N = length(fields) #number of MADE layers
x_symbols = vcat([:x], [gensym() for _ in 1:N])
total_std = [gensym() for _ in 1:N]
st_symbols = [gensym() for _ in 1:N]
calls1 = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]
calls2 = [:($(x_symbols[i]) = coord_transform_smooth($(x_symbols[i-1]),$(x_symbols[i]))) for i in 2:N]
calls3 = [:($(total_std[i]) = copy($(x_symbols[i+1]))) for i in 1:N]



n = length(calls1) + length(calls2) + length(calls3)
calls = similar(calls1, n)

#add up all the log std for each layer
#each_layer_ouptut = :([$(x_symbols[N]) for i in 1:N])

################# add the definition of total_std as an array in the list of blocks
#calls[1] .= :($(x_symbols[1]) = 1)
calls[1:3:n] .= calls1
calls[2:3:n] .= calls3
calls[3:3:n] .= calls2


push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st, $(x_symbols[N]), $(total_std...)))
return Expr(:block, calls...)
end


#The sample function for the MAF
#TODO Also needs to verify this is not causing the bug
function sample(T::MAF, ps, st; specific_sample = randn(T.layers[1].layers[1].in_dims), debug = false)
  _sample = specific_sample
  for i in reverse(eachindex(T.layers))
    _sample = sample(T.layers[i], ps[i], st[i], samples = _sample, use_softplus = T.softplus)
    debug && @debug "MAF sampling layer" layer=i sample=_sample
  end
  return _sample
end