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

model = MAF(model1,model2, model3, softplus=true)

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

tstate = main(tstate, vjp_rule, train_dataloader, 1500)

testx = [0.5,0.5,0.5,0.5]
y, st, x1, x2...  = Lux.apply(model, testx, tstate.parameters, tstate.states)

y_pred = y
data = x1
extra = x2

u = forward(data, y_pred)

d = u
for i in Iterators.reverse(x2)
    d = inverse(d, i)
    print(d)
end

specific_sample = Sbi.sample(tstate.model, tstate.parameters, tstate.states, specific_sample = u, debug=true)

samples = [Sbi.sample(tstate.model, tstate.parameters, tstate.states) for i in 1:10000]
result = hcat(samples...)'

#Plot Samples from the true distribution
using CairoMakie

# A better subplot

f = Figure()

ax = Axis(f[1,1])
ax2 = Axis(f[1,2])


generated_data = first(train_dataloader)'

scatter!(ax, generated_data[:,1], generated_data[:,2], rasterize = true, color=:blue)
scatter!(ax, result[:,1], result[:,2], rasterize = true, color=:red)


save("comb.png", f)

data = first(train_dataloader)

y_pred, st, x1, x2...  = Lux.apply(model, data, tstate.parameters, tstate.states)

print(size(y_pred),size(data))
#print(data)
#println(n, half1, half2)
u = forward(x1, y_pred)

y_pred, st, x1, x2...  = Lux.apply(model, result', tstate.parameters, tstate.states)
u2 = forward(x1, y_pred)



#Now I'm going to plot the transformed distributions to see if one looks more like a standard normal distribution than another


f_standard = Figure();

ax = Axis(f_standard[1,1])
ax2 = Axis(f_standard[1,2])



scatter!(ax, u'[:,1], u'[:,2], rasterize = true, color=:blue)
scatter!(ax2, u2'[:,1], u2'[:,2], rasterize = true, color=:red)


save("comb_normal.png", f_standard)