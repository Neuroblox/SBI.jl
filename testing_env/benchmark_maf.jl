using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi
using CairoMakie


import MLUtils: DataLoader, splitobs
include("../src/utils.jl")

# This generates the needed Data
function generate_data_simple_nonlinear(n, batch_size)
    class1 = (randn(n) * 1.3)
    class2 = (randn(n) .+ (1/4) .* (class1 .^ 2))
    class3 = (randn(n) .+ (1/4) .* (class2 .^ 2))
    class4 = (randn(n) .+ (1/4) .* (class3 .^ 2))
    class5 = (randn(n) .+ (1/4) .* (class4 .^ 2))
    class6 = (randn(n) .+ (1/4) .* (class5 .^ 2))

    data = vcat(class2', class1', class3', class4')
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

# Generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 12345)

# Set the optimizer model
opt = Adam(0.060)

# Define the models
function create_model(MADE_layer, softplus)
    if MADE_layer == MADE
        model1 = MADE(MaskedLinear(4, 20, relu), MaskedLinear(20, 20, relu), MaskedLinear(20, 20, relu), MaskedLinear(20, 8))
        model2 = MADE(MaskedLinear(4, 20, relu), MaskedLinear(20, 20, relu), MaskedLinear(20, 20, relu), MaskedLinear(20, 8), random_order=true)
        model3 = MADE(MaskedLinear(4, 20, relu), MaskedLinear(20, 20, relu), MaskedLinear(20, 20, relu), MaskedLinear(20, 8), random_order=true)
    else
        model1 = MADE_relu(4, 20)
        model2 = MADE_relu(4, 20, random_order=true)
        model3 = MADE_relu(4, 20, random_order=true)
    end
    return MAF(model1, model2, model3, softplus=softplus)
end


#other benchmark model that we know works sanity check
model = MADE(MaskedLinear(4, 50, relu), MaskedLinear(50, 50, relu), MaskedLinear(50, 8))
model2 = MADE(MaskedLinear(4, 50, relu), MaskedLinear(50, 50, relu), MaskedLinear(50, 8), random_order=true)
model3 = MADE(MaskedLinear(4, 50, relu), MaskedLinear(50, 50, relu), MaskedLinear(50, 8), random_order=true)
model = MAF(model, model2, model3, softplus=true)

# Create models with different configurations
models = [
    #model,
    #create_model(MADE, true),
    #create_model(MADE, false),
    create_model(MADE_relu, true),
    #create_model(MADE_relu, false)
]

# Function to run training
function run_training(model, rng, opt, epochs)
    ps, st = Lux.setup(rng, model)
    tstate = Lux.Training.TrainState(model, ps, st, opt)
    train_dataloader = generate_data_simple_nonlinear(100000, 60000)
    vjp_rule = AutoZygote()

    function main(tstate::Lux.Training.TrainState, vjp, data_loader, epochs)
        for epoch in 1:epochs
            for data in data_loader
                grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, lux_gaussian_maf_loss, data, tstate)
                println("Epoch: $(epoch) || Loss: $(loss)")
                tstate = Lux.Training.apply_gradients(tstate, grads)
            end
        end
        return tstate
    end

    return main(tstate, vjp_rule, train_dataloader, epochs)
end

# Run training for each model
epochs = 400
trained_states = [run_training(model, rng, opt, epochs) for model in models]

train_dataloader = generate_data_simple_nonlinear(100000, 60000)
generated_data = first(train_dataloader)'

# Generate samples and plot results
for (i, tstate) in enumerate(trained_states)
    # Generate samples
    samples = [Sbi.sample(tstate.model, tstate.parameters, tstate.states) for _ in 1:100000]
    result = hcat(samples...)'

    # Plot samples from the true distribution
    f = Figure()
    ax = Axis(f[1, 1])
    ax2 = Axis(f[1, 2])
    scatter!(ax, generated_data[:, 1], generated_data[:, 2], rasterize=true, color=:blue)
    scatter!(ax2, result[:, 1], result[:, 2], rasterize=true, color=:red)
    save("model_sample_$i.png", f)

    data = first(train_dataloader)

    y_pred, st, x1, x2... = Lux.apply(tstate.model, data, tstate.parameters, tstate.states)
    u = forward(x1, y_pred)

    y_pred, st, x1, x2... = Lux.apply(tstate.model, result', tstate.parameters, tstate.states)
    u2 = forward(x1, y_pred)

    # Plot the transformed distributions
    f_standard = Figure()
    ax = Axis(f_standard[1, 1])
    ax2 = Axis(f_standard[1, 2])
    scatter!(ax, u'[:, 1], u'[:, 2], rasterize=true, color=:blue)
    scatter!(ax2, u2'[:, 1], u2'[:, 2], rasterize=true, color=:red)
    save("model_transform_$i.png", f_standard)
end

models[1].softplus

model = models[1]


# Inverse Testing
testx = rand(4,5)


model = create_model(MADE_relu, true)
ps, st = Lux.setup(rng, model)
tstate = Lux.Training.TrainState(model, ps, st, opt)


y, st, x1, x2...  = Lux.apply(model, testx, tstate.parameters, tstate.states)

y_pred = y
data = x1
extra = x2

sum_output = sum(log.(softplus.(-i)) for i in extra)

n = div(size(y_pred)[1], 2)
half1 = @view y_pred[1:n,:]
half2 = @view y_pred[n+1:end,:]

half2_all = @view sum_output[n+1:end,:]
println(sum(mean(half2_all, dims=2)))
# ------------------------------------------ THIS IS THE BUG ---------------------------------------
u = forward(data, y_pred)

println(u)

#invert the amf transformation back to the original
#NOTE x2 contains y_pred
d = u
for i in Iterators.reverse(x2)
    d = inverse(d, i)
end