using JLD2


function softplus(x::Number; β::Number = 1.0, threshold::Number = 20, ϵ::Number = 1e-3)
    if x*β > threshold
        return β * x + ϵ
    else
        return (1 / β ) * log(1 + exp( β * x)) + ϵ
    end
end

#divides along the rows in 2 for made_output
# assumes each column of u is a different data point
function inverse(u, made_output)
    n = div(size(made_output)[1], 2)
    half1 = @view made_output[1:n,:]
    half2 = @view made_output[n+1:end,:]
    return u .* softplus.(half2) + half1
end

function forward(x, made_output)
    n = div(size(made_output)[1], 2)
    half1 = @view made_output[1:n,:]
    half2 = @view made_output[n+1:end,:]
    return (x.-half1)./softplus.(half2)
end


function save_model(tstate, name)
    save_object("$name.jld2", tstate)
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