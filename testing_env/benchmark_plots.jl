using CSV
using DataFrames
using CairoMakie

# Read the CSV files
df1 = CSV.read("losses_res_relu.csv", DataFrame)
df2 = CSV.read("losses_res_soft.csv", DataFrame)
df3 = CSV.read("losses_normal_relu.csv", DataFrame)
df4 = CSV.read("losses_normal_soft.csv", DataFrame)

# Extract the epoch and loss values, leaving out the first 10
epochs1 = df1.epoch[11:end]
losses1 = df1.loss[11:end]

epochs2 = df2.epoch[11:end]
losses2 = df2.loss[11:end]

epochs3 = df3.epoch[11:end]
losses3 = df3.loss[11:end]

epochs4 = df4.epoch[11:end]
losses4 = df4.loss[11:end]

# Create the plot
fig = Figure(resolution = (800, 600))

ax = Axis(fig[1, 1], title = "Loss Functions", xlabel = "Epoch", ylabel = "Loss")

lines!(ax, epochs1, losses1, label = "Res ReLU", color = :blue)
lines!(ax, epochs2, losses2, label = "Res Soft", color = :red)
lines!(ax, epochs3, losses3, label = "Normal ReLU", color = :green)
lines!(ax, epochs4, losses4, label = "Normal Soft", color = :purple)

axislegend(ax)

# Save the figure as benchmark.png
save("benchmark.png", fig)