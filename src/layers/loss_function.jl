using Lux, Optimisers, Random, Zygote
include("../utils.jl")

function log_std_loss(y_pred, data)
    #print(size(y_pred),size(data))
    #print(data)
    n = div(size(y_pred)[1], 2)
    #data = data[1:n]


    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]

    #println(n, half1, half2)
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

    #println(extra)
    #println("logstd_loss2_called")
    #println(size(y_pred),size(data))
    #print(data)
    n = div(size(y_pred)[1], 2)
    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]

    half2_all = @view sum_output[n+1:end,:]
    #println(y_pred)
    #println(n, half1, half2)

    #print("This is the set of variances fo fuck rith offf   ")
    #println(half2)
    #print("this is data")
    #println(data)
    #println("this is the first half")
    #println(half1)

    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    u = (data.-half1).*exp.(-half2)
    #println(u)
    #println("This is right before I need it")
    #println(size(u), size(half2_all))
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2) .+ half2_all

    #println("debug negloglike $negloglike")

    negloglike = mean(negloglike, dims=2)
    negloglike = sum(negloglike)
    if (negloglike == Inf) 
        DomainError(val) 
    end
    #println("about to return negloklike")
    return negloglike
end

function log_std_loss2_smooth(y_pred, data, extra)
    sum_output = sum(log.(softplus.(i)) for i in extra)

    n = div(size(y_pred)[1], 2)
    half2_all = @view sum_output[n+1:end,:] # note do I add the 1e-3
    #println(sum(mean(half2_all, dims=2)))
    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    #println("y_pred", y_pred)
    u = forward(data, y_pred)
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2)
    loglike = -negloglike
    scale = softplus.(half2_all) .+ 1e-3
    logscale = log.(scale)
    #println("before det", loglike)
    #println("half2_all", half2_all)
    #println("loglike", loglike)
    #println("logscale", logscale)
    loglike = loglike + half2_all
    #println("after det", half2_all)
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
    #println(sum(mean(half2_all, dims=2)))
    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    #println("y_pred", y_pred)
    u = forward(data, y_pred)

    # Note the next line is wrong
    # Need to look at how the context is stored
    #u = conditional_forward(u, y_pred)
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2)
    loglike = -negloglike
    scale = softplus.(half2_all) .+ 1e-3
    logscale = log.(scale)
    #println("before det", loglike)
    #println("half2_all", half2_all)
    #println("loglike", loglike)
    #println("logscale", logscale)
    loglike = loglike + half2_all
    #println("after det", half2_all)
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
    #println("loss function called")
    y, st, x1, x2...  = Lux.apply(model, data, ps, st)
    #println("x1", x1)
    #println("x2", x2)
    #println(size(x2))
    if model.softplus
        loss = log_std_loss2_smooth(y, x1, x2) #TODO double check this
        println("using softmax loss")
    else
        loss = log_std_loss2(y, x1, x2) #TODO double check thiso
        println("using relu loss")
    end
    return loss, st, ()
end

# New loss function to work with the improved interface
function logp_conditional_maf_smooth(output, st)
    sum_output = sum(i for i in [st.encoder_output])
    sum_output2 = sum(i for i in [st.MADE_output])

    n = size(output)[1]
    half2_all = @view sum_output[n+1:end,:] # note do I add the 1e-3
    half2_all2 = @view sum_output2[n+1:end,:] # note do I add the 1e-3
    #println(sum(mean(half2_all, dims=2)))
    # ------------------------------------------ THIS IS THE BUG ---------------------------------------
    #println("y_pred", y_pred)

    # Note the next line is wrong
    # Need to look at how the context is stored
    #u = conditional_forward(u, y_pred)
    negloglike = 0.5.*(output.^2)
    loglike = -negloglike
    println(loglike)
    #println("before det", loglike)
    println("half2_all", half2_all)
    #println("loglike", loglike)
    #println("logscale", logscale)
    loglike = loglike .- half2_all
    println(loglike)
    loglike = loglike .- 0.5*log(2*pi)
    println(loglike)

    #println("after det", half2_all)
    loglike = sum(loglike)
    #is this worth it? I'm not sure, 
    #if (negloglike == Inf) 
    #    DomainError(val) 
    #end
    logabsdet = sum(log.(softplus.(half2_all2)) .+ 1e-3)
    println("half2all2", logabsdet)
    return loglike + logabsdet
end