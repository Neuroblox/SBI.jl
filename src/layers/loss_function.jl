using Lux, Optimisers, Random, Zygote
include("../utils.jl")

function log_std_loss(y_pred, data)
    #print(size(y_pred),size(data))
    #print(data)
    n = div(size(y_pred)[1], 2)
    #data = data[1:n]


    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]

    #@debug "log_std_loss" n=n half1=half1 half2=half2
    u = (data.-half1).*exp.(-half2)
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2) .+ half2
    negloglike = mean(negloglike, dims=2)
    negloglike = sum(negloglike)
    #if (negloglike == Inf) 
     #   DomainError(val) 
    #:wend
    return negloglike
end



function log_std_loss2(y_pred, data, extra)
    sum_output = sum(extra)

    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("log_std_loss2 called: extra=$extra, y_pred_size=$(size(y_pred)), data_size=$(size(data))")
    end
    n = div(size(y_pred)[1], 2)
    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]

    half2_all = @view sum_output[n+1:end,:]
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("log_std_loss2 intermediate values: y_pred=$y_pred, n=$n, half1=$half1, half2=$half2")
        println("Variance values: half2=$half2")
        println("Data values: data=$data")
        println("First half values: half1=$half1")
    end

    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    u = (data.-half1).*exp.(-half2)
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("Before negloglike calculation: u=$u, half2_all_size=$(size(half2_all))")
    end
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2) .+ half2_all

    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("Calculated negloglike: negloglike=$negloglike")
    end

    negloglike = mean(negloglike, dims=2)
    negloglike = sum(negloglike)
    if (negloglike == Inf) 
        DomainError(val) 
    end
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("About to return negloglike")
    end
    return negloglike
end

function log_std_loss2_smooth(y_pred, data, extra)
    sum_output = sum(log.(softplus.(i)) for i in extra)

    n = div(size(y_pred)[1], 2)
    half2_all = @view sum_output[n+1:end,:] # note do I add the 1e-3
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("log_std_loss2_smooth: sum_mean_half2_all=$(sum(mean(half2_all, dims=2)))")
        println("y_pred values: y_pred=$y_pred")
    end
    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    u = forward(data, y_pred)
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2)
    loglike = -negloglike
    scale = softplus.(half2_all) .+ 1e-3
    logscale = log.(scale)
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("Before determinant calculation: loglike=$loglike")
        println("Intermediate values: half2_all=$half2_all, loglike=$loglike, logscale=$logscale")
    end
    loglike = loglike + half2_all
    if haskey(ENV, "JULIA_DEBUG") && ENV["JULIA_DEBUG"] == "sbi"
        println("After determinant calculation: half2_all=$half2_all")
    end
    loglike = mean(loglike, dims=2)
    loglike = sum(loglike)
    #if (negloglike == Inf) 
    #    DomainError(val) 
    #end
    return -loglike
end

function log_conditional_maf_smooth(y_pred, data, extra)
    sum_output = sum(log.(softplus.(i)) for i in extra)

    n = div(size(y_pred)[1], 2)
    half2_all = @view sum_output[n+1:end,:] # note do I add the 1e-3
    #@debug "log_conditional_maf_smooth" sum_mean_half2_all=sum(mean(half2_all, dims=2))
    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    #@debug "y_pred values" y_pred=y_pred
    u = forward(data, y_pred)

    # Note the next line is wrong
    # Need to look at how the context is stored
    #u = conditional_forward(u, y_pred)
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2)
    loglike = -negloglike
    scale = softplus.(half2_all) .+ 1e-3
    logscale = log.(scale)
    #@debug "Before determinant calculation" loglike=loglike
    #@debug "Intermediate values" half2_all=half2_all loglike=loglike logscale=logscale
    loglike = loglike + half2_all
    #@debug "After determinant calculation" half2_all=half2_all
    loglike = mean(loglike, dims=2)
    loglike = sum(loglike)
    #if (negloglike == Inf) 
    #    DomainError(val) 
    #end
    return -loglike
end

function log_MAF_loss(u)
    n = length(u)
    mu = (1/n)*sum(u)
    sigma = (1/n)*sum((mu .- u).^2)

    return (mu^2 + (sigma -1)^2)
end


# for lux.jl loss function needts to take 4 parameter,  and return 3 parameters

# input: model, parameters, states and data.

# output: loss, updated_state, and any computed statistics

function lux_gaussian_made_loss(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    loss = log_std_loss(y_pred, data)
    return loss, st, ()
end

function lux_gaussian_maf_loss(model, ps, st, data)
    #@debug "Loss function called"
    y, st, x1, x2...  = Lux.apply(model, data, ps, st)
    #@debug "Model applied" x1=x1 x2=x2 x2_size=size(x2)
    if model.softplus
        loss = log_std_loss2_smooth(y, x1, x2) #TODO double check this
        #@debug "Using softmax loss"
    else
        loss = log_std_loss2(y, x1, x2) #TODO double check thiso
        #@debug "Using relu loss"
    end
    return loss, st, ()
end

# New loss function to work with the improved interface
#not exact but close enough for now
function logp_conditional_maf_smooth(output, st)
    output = inverse_exp(output, st.encoder_output)
    sum_output = sum(i for i in [st.encoder_output])
    sum_output2 = sum(i for i in [st.MADE_output])

    n = size(output)[1]
    half2_all = @view sum_output[n+1:end,:] # note do I add the 1e-3
    half2_all2 = @view sum_output2[n+1:end,:] # note do I add the 1e-3
    #@debug "logp_conditional_maf_smooth" sum_mean_half2_all=sum(mean(half2_all, dims=2))
    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    #@debug "y_pred values" y_pred=y_pred

    # Note the next line is wrong
    # Need to look at how the context is stored
    #u = conditional_forward(u, y_pred)
    negloglike = 0.5.*(output.^2)
    loglike = -negloglike
    #@debug "Initial loglike" loglike=loglike
    #@debug "Before determinant calculation" half2_all=half2_all
    loglike = loglike .- half2_all
    #@debug "After subtracting half2_all" loglike=loglike
    loglike = loglike .- 0.5*log(2*pi)
    #@debug "After subtracting log(2π) term" loglike=loglike

    loglike = sum(loglike)
    #is this worth it? I'm not sure, 
    #if (negloglike == Inf) 
    #    DomainError(val) 
    #end
    logabsdet = sum(log.(softplus.(half2_all2)) .+ 1e-3)
    #@debug "Log absolute determinant" logabsdet=logabsdet
    return loglike + logabsdet
end