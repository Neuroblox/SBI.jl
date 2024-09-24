using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi

import MLDatasets: MNIST

import MLUtils: DataLoader, splitobs



#This generate the needed Data,
function generate_data_simple_nonlinear(n,batch_size)
    class1 = randn(n)*2
    class2 = randn(n) .+ (1/4).*(class1.^2)

    data = vcat(class1',class2')
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

#generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 12345)

# set the optimiser model
opt = Adam(0.060)

model = MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4))
model2 = MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model3 = MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model = MAF(model, model2, model3)

tstate = Lux.Training.TrainState(rng, model, opt);

train_dataloader = generate_data_simple_nonlinear(40000,5000)

vjp_rule = Lux.Training.AutoZygote()
ADTypes.AutoZygote()

function main(tstate::Lux.Training.TrainState, vjp, data_loader, epochs)
    for epoch in 1:epochs
        for data in data_loader
            grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
            lux_gaussian_maf_loss, data, tstate)
            println("Epoch: $(epoch) || Loss: $(loss)")
            tstate = Lux.Training.apply_gradients(tstate, grads)
        end
    end
    return tstate
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstate = main(tstate, vjp_rule, train_dataloader, 1000)


#Plot Samples from the true distribution

using CairoMakie

# A better subplot

f = Figure()

ax = Axis(f[1,1], limits = (-10, 10, -5, 15))
ax2 = Axis(f[1,2], limits = (-10, 10, -5, 15))

scatter!(ax, generated_data[:,1], generated_data[:,2], rasterize = true, color=:blue)
scatter!(ax2, result[:,1], result[:,2], rasterize = true, color=:red)


save("comb.png", f)
