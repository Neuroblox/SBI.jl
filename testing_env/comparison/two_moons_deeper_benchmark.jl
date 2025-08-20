using Pkg
Pkg.activate("./testing_env")
using Revise

using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi
using CairoMakie

import MLUtils: DataLoader, splitobs
include("../../src/utils.jl")

using JSON3

# filepath: /home/simon/Code/SBI.jl/testing_env/twoMoons.json
file_path = "/home/simon/Code/SBI.jl/testing_env/twoMoons.json"

# Read and parse the JSON file
data = JSON3.read(open(file_path))

# Display the data
println(data)

matrix = hcat(data...)

# Generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 12345)

# Set the optimizer model
opt = Adam(0.060)

model1 = MADE_relu(2, 4, order_permutation=1)
model2 = MADE_relu(2, 4, order_permutation=0)
model3 = MADE_relu(2, 4, order_permutation=1)
model4 = MADE_relu(2, 4, order_permutation=0)
model5 = MADE_relu(2, 4, order_permutation=1)
simple_model = [MAF(model1, model2, model3, model4, model5, softplus=true)]

using Dates
using FilePathsBase

# Create a "data" folder if it doesn't exist
data_folder = "data"
if !isdir(data_folder)
    mkdir(data_folder)
end

# Replace run_id with the current date and time
function run_training(model, rng, opt, epochs, data)
    ps, st = Lux.setup(rng, model)
    tstate = Lux.Training.TrainState(model, ps, st, opt)
    train_dataloader = data
    vjp_rule = AutoZygote()

    function main(tstate::Lux.Training.TrainState, vjp, data_loader, epochs)
        # Create a unique filename prefix using the current date and time
        run_id = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        loss_file_path = joinpath(data_folder, "losses_$run_id.csv")
        loss_file = open(loss_file_path, "w")
        println(loss_file, "epoch,loss")

        for epoch in 1:epochs
            for (i, data) in enumerate(data_loader)
                grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, lux_gaussian_maf_loss, data, tstate)
                println("Epoch: $(epoch) || Loss: $(loss)")
                println(loss_file, "$epoch,$loss")

                if i % 50 == 0
                    ps_file = joinpath(data_folder, "tstate_ps_$run_id_epoch_$(epoch)_iter_$(i).jls")
                    st_file = joinpath(data_folder, "tstate_st_$run_id_epoch_$(epoch)_iter_$(i).jls")
                    serialize(ps_file, tstate.ps)
                    serialize(st_file, tstate.st)
                end

                tstate = Lux.Training.apply_gradients(tstate, grads)
            end
        end

        close(loss_file)
        return tstate
    end

    return main(tstate, vjp_rule, train_dataloader, epochs)
end

train_dataloader = DataLoader(matrix; batchsize=256, shuffle=true)
epochs = 1000
trained_states = [run_training(model, rng, opt, epochs, train_dataloader) for model in simple_model]

generated_data = first(train_dataloader)'

# Generate samples and plot results
for (i, tstate) in enumerate(trained_states)
    # Generate samples
    samples = [Sbi.sample(tstate.model, tstate.parameters, tstate.states) for _ in 1:250]
    result = hcat(samples...)'

    # Plot samples from the true distribution
    f = Figure()
    ax = Axis(f[1, 1])
    ax2 = Axis(f[1, 2])
    scatter!(ax, generated_data[:, 1], generated_data[:, 2], rasterize=true, color=:blue)
    scatter!(ax2, result[:, 1], result[:, 2], rasterize=true, color=:red)
    save(joinpath(data_folder, "model_sample_2m_$i.png"), f)

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
    save(joinpath(data_folder, "model_transform_2m_$i.png"), f_standard)
end