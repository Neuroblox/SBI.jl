
function softplus(x::Number; β::Number = 1.0, threshold::Number = 20, ϵ::Number = 1e-3)
    if x*β > threshold
        return β * x + ϵ
    else
        return (1 / β ) * log(1 + exp( β * x)) + ϵ
    end
end


#=
using Plots

# Code that only runs if utils.jl is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Plots

    x_values = -100:0.1:100

    # Compute softplus values for each x
    y_values = [softplus(x) for x in x_values]

    # Plot the softplus function
    plot(x_values, y_values, label="softplus(x)", xlabel="x", ylabel="softplus(x)", title="Softplus Function")

    # Save the plot as a PNG file
    savefig("softplus_function.png")
end
=#

