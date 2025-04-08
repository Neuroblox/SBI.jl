using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays

using Sbi


import MLUtils: DataLoader, splitobs

include("../src/utils.jl")



#This generate the needed Data,
function generate_data_simple_nonlinear(n,batch_size)
    class1 = (randn(n) * 1.3)
    class2 = (randn(n) .+ (1/4).*(class1.^2))
    class3 = (randn(n) .+ (1/4).*(class2.^2))
    class4 = (randn(n) .+ (1/4).*(class3.^2))
    class5 = (randn(n) .+ (1/4).*(class4.^2))
    class6 = (randn(n) .+ (1/4).*(class5.^2))
    num_iter = 10000
    for i in range(num_iter):
        x, y = datasets.make_moons(128, noise=.1)
        x = torch.tensor(x, dtype=torch.float32)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 500 == 0:
            xline = torch.linspace(-1.5, 2.5, steps=100)
            yline = torch.linspace(-.75, 1.25, steps=100)
            xgrid, ygrid = torch.meshgrid(xline, yline)
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    
            with torch.no_grad():
                zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)
    
            plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
            plt.title('iteration {}'.format(i + 1))
            plt.show()
    data = vcat(class2',class1', class3', class4') 
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

#generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 123456)

# set the optimiser model
opt = Adam(0.060)

model = MADE(MaskedLinear(4,20), MaskedLinear(20,20), MaskedLinear(20,20), MaskedLinear(20,8))
model2 = MADE(MaskedLinear(4,20), MaskedLinear(20,20), MaskedLinear(20,20), MaskedLinear(20,8), random_order=true)
model3 = MADE(MaskedLinear(4,20), MaskedLinear(20,20), MaskedLinear(20,20), MaskedLinear(20,8), random_order=true)
model = MAF(model, model2, model3, softplus=true)

ps, st = Lux.setup(rng, model)

tstate = Lux.Training.TrainState(model, ps, st, opt);

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

#generate samples

samples = [Sbi.sample(tstate.model, tstate.parameters, tstate.states) for i in 1:100000]
result = hcat(samples...)'

#Plot Samples from the true distribution
using CairoMakie

# A better subplot

f = Figure()

ax = Axis(f[1,1])
ax2 = Axis(f[1,2])


generated_data = first(train_dataloader)'

scatter!(ax, generated_data[:,1], generated_data[:,2], rasterize = true, color=:blue)
scatter!(ax2, result[:,1], result[:,2], rasterize = true, color=:red)


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



testx = rand(2,5)

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

a = [0.005526502695354485 0.9203793437239806 0.8583302331427589 0.7263882348107662 0.8566019782063469; 0.7683213522232963 0.1234125315921778 0.924756973471102 0.5787676213861526 0.27316396167837376]
b = [0.286298930644989 0.286298930644989 0.286298930644989 0.286298930644989 0.286298930644989; 3.1493522522644426 1.7465290240106406 1.6959080480579325 1.6484566362895738 1.6944981011504456; 0.5162094831466675 0.5162094831466675 0.5162094831466675 0.5162094831466675 0.5162094831466675; 57.6576934436217 74.2537316540198 71.75461840056322 66.92630817205031 71.68501055644616]
c = [-0.2849909898672825 0.6436073723423182 0.580625983556882 0.44670157728844634 0.5788717618597308; -0.04129526282743994 -0.021858761809027356 -0.010746908629259025 -0.01598284831885913 -0.019827217729641987]

e = [-0.2849909898672825 0.6436073723423182 0.580625983556882 0.44670157728844634 0.5788717618597308; 
     -0.04129526282743994 -0.021858761809027356 -0.010746908629259025 -0.01598284831885913 -0.019827217729641987]
f = [0.22038879990577698 0.22038879990577698 0.22038879990577698 0.22038879990577698 0.22038879990577698; 
     2.671335771447992 -0.34287419857908774 -0.1904154372588902 0.15854954251971698 -0.1858620982559347; 
     0.5248976349830627 0.5248976349830627 0.5248976349830627 0.5248976349830627 0.5248976349830627; 
     67.43040250339445 50.7468436579165 51.22930666997978 52.74767368486531 51.24818989911339]
g = [-0.510151008557262 0.4272141188424223 0.36363813171469145 0.22844936415642647 0.36186734865976045; 
     -0.04022800851782492 0.00632569610117775 0.003507075016884768 -0.003308754109748414 0.0032397561962079933]