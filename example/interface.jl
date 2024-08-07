using Turing
using MLUtils


# testing sampling from a  turing model
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    b² ~ InverseGamma(2, 3)
    x ~ Normal(m, sqrt(b²))
    y ~ Normal(m, sqrt(s²))
    return x, y
end

g_prior_sample = gdemo(missing, missing)

sample(g_prior_sample, Prior(), 10)



rand(g_prior_sample)


### example of advi interface so we can make this more compatable
## taken from https://turing.ml/dev/tutorials/09-variational-inference/

# generate data
x = randn(2000);

@model function model_vi(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end;

# Instantiate model
m = model_vi(x);

advi = ADVI(10, 1000)
q = vi(m, advi);

#define function that creates a dataloader from a turing model
    #m - model
    #batchsize - the batchsize for the dataloader
    # total number of samples
    # parameters - array of symbols letting me know which are parameters
    # prior - array of symbols denoting which are parameters
    # output_data - array of symbols denoting which are output_data
function generate(m, batchsize, total, parameters, output_data)
    #create an array to store the total numer of data
    parameter_length, data_length = length(parameters), length(output_data)
    total_length = parameter_length + data_length

    data = Array{Float64}(undef, total, total_length)

    for i in 1:total
        data[i,1:parameter_length] = collect(values(rand(m(fill(missing, parameter_length)...))[parameters]))
        data[i,parameter_length+1:end] = collect(values(rand(m(fill(missing, data_length)...))[output_data]))
    end

    data_loader = DataLoader(data'; batchsize=2, shuffle=true)

    return data_loader

end