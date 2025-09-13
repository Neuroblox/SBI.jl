using Logging

function softplus(x::Number; β::Number = 1.0, threshold::Number = 20, ϵ::Number = 1e-3)
    if x*β > threshold
        return β * x + ϵ
    else
        return (1 / β ) * log(1 + exp( β * x)) + ϵ
    end
end

#divides along the rows in 2 for made_output
# assumes each column of u is a different data point
function forward(u, made_output)
    n = div(size(made_output)[1], 2)
    half1 = @view made_output[1:n,:]
    half2 = @view made_output[n+1:end,:]
    @debug "Forward function inputs" u=u made_output=made_output n=n half1=half1 half2=half2
    logstd_sum = sum(log.(softplus.(half2).+ 1e-3))
    @debug "Forward function logstd calculation" logstd=logstd_sum
    result = u .* softplus.(half2) + half1
    @debug "Forward function result" result=result
    return result
end

# add epsilon, check if I didnt mess up forward and inverse
# I did, what is the justification
function inverse(x, made_output)
    n = div(size(made_output)[1], 2)
    half1 = @view made_output[1:n,:]
    half2 = @view made_output[n+1:end,:]
    @debug "Inverse function" x=x made_output=made_output n=n half1=half1 half2=half2
    result = (x.-half1)./softplus.(half2)
    @debug "Inverse function result" result=result
    return result
end


function inverse_exp(x, made_output)
    n = div(size(made_output)[1], 2)
    @debug "Inverse exponential function" x=x made_output=made_output n=n
    half1 = @view made_output[1:n,:]
    half2 = @view made_output[n+1:end,:]
    @debug "Inverse exponential components" half1=half1 half2=half2
    result = (x.-half1)./exp.(half2)
    @debug "Inverse exponential result" result=result
    return result
end

function save_model(tstate, name)
    save_object("$name.jld2", tstate)
end

# Conditional Forward Function
# forward function assuming the conditional base distribution
# the means and standard deviations of the gaussian are given  as input
# NOTE: currently Untested
# half2 is logstd
function conditional_forward(u, context)
    n = div(size(context)[1], 2)
    half1 = @view context[1:n,:]
    half2 = @view context[n+1:end,:]
    @debug "Conditional forward function" u=u context=context n=n half1=half1 half2=half2
    result = (u .- half1).*exp.(-context[2,:])
    @debug "Conditional forward result" result=result
    return result
end

# removes context and applies the forward mode of the function
function conditional_forward_split(u, made_output, context)
    u_x = u[1:end-context]
    @debug "Conditional forward split" u=u made_output=made_output context=context u_x=u_x
    result = forward(u_x, made_output)
    @debug "Conditional forward split result" result=result
    return result
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