using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi
using CairoMakie


import MLUtils: DataLoader, splitobs
include("../src/utils.jl")


function generate_data_simple_nonlinear(n,batch_size)
    class1 = (randn(n) * 1.3)
    class2 = (randn(n) .+ (1/4).*(class1.^2))
    class3 = (randn(n) .+ (1/4).*(class2.^2))
    class4 = (randn(n) .+ (1/4).*(class3.^2))
    class5 = (randn(n) .+ (1/4).*(class4.^2))
    class6 = (randn(n) .+ (1/4).*(class5.^2))

    data = vcat(class2',class1', class3', class4') 
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end


rng = MersenneTwister()
Random.seed!(rng, 12345)

# Set the optimizer model
opt = Adam(0.060)

model1 = MADE_relu(4, 20)
model2 = MADE_relu(4, 20, random_order=true)
model3 = MADE_relu(4, 20, random_order=true)

model = MAF(model1, model2, model3, softplus=true)

testx = rand(4,5)

ps, st = Lux.setup(rng, model)
tstate = Lux.Training.TrainState(model, ps, st, opt)

train_dataloader = generate_data_simple_nonlinear(100000,60000)

vjp_rule = AutoZygote()


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

tstate = main(tstate, vjp_rule, train_dataloader, 500)
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

#check if the sample function is consistent

u_samples = randn(4, 100000)

specific_sample = [Sbi.sample(tstate.model, tstate.parameters, tstate.states, specific_sample = u_samples[:,i]) for i in 1:100000]

matrix = hcat(specific_sample...)


#Now go forward again
y, st, x1, x2...  = Lux.apply(model, matrix, tstate.parameters, tstate.states)

u = forward(x1, y)

sample_error = abs.(u .- u_samples)

maximum(sample_error)

using CairoMakie

f_standard = Figure();

ax = Axis(f_standard[1,1])
ax2 = Axis(f_standard[1,2])



scatter!(ax, u'[:,1], u'[:,2], rasterize = true, color=:blue)
scatter!(ax2, u_samples'[:,1], u_samples'[:,2], rasterize = true, color=:red)


save("comb_normal_2.png", f_standard)


