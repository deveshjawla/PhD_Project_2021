using Statistics, Turing, Plots, DataFrames, DelimitedFiles, ReverseDiff, Printf, BSON

#### 
#### Network - A customizable chain of parametrized conv layer and dense layers. Does not require manually
#### calculating the output size of images. Takes as input batches of images with dims WHCN.
#### Automatically makes the weight matrices which are to be modified by Variational Inference.
#### 

using Flux, Parameters:@with_kw

@with_kw struct DenseParams
    indim::Int = 128
    outdim::Int = 128
    activation_fn = relu
    bnmom::Union{Float32,Nothing} = nothing
    bias::Bool = false
end

"""
# Function to make a custom dense layer
"""
function layer(weight::AbstractArray, bias::AbstractArray, dp::DenseParams)
    if dp.bnmom === nothing
        Dense(weight, bias, dp.activation_fn)
    else
        Chain(
            Dense(weight, bias),
            BatchNorm(dp.outdim, dp.activation_fn, momentum = dp.bnmom)
        )
    end
end

@with_kw struct ConvParams
    filter_size::Tuple{Int,Int} = (3, 3)
    in_channels::Int = 1
    out_channels::Int = 1
    activation_fn = relu
    stride_length::Int = 1
    pad::Int = 1
    pool_window::Tuple{Int,Int} = (2, 2)
    pool_stride::Int = 1
    bnmom::Union{Float32,Nothing} = nothing
end

"""
# Fucntion to make a custom Covolution layer.
## Returns a tuple, (layer, output_image_dims)
"""
function layer(weight::AbstractArray, bias::AbstractArray, cp::ConvParams)
    if cp.bnmom === nothing
        layer = Chain(
            Conv(weight, bias, cp.activation_fn, pad = cp.pad),
            x -> meanpool(x, cp.pool_window, stride = cp.pool_stride)
        )
    else
        layer = Chain(
            Conv(weight, bias, pad = cp.pad),
            x -> meanpool(x, cp.pool_window, stride = cp.pool_stride),
            BatchNorm(cp.in_channels, cp.activation_fn, momentum = cp.bnmom)
        )
    end
    return layer
end

Turing.setadbackend(:reversediff)
using Turing.Variational
gr()



function conv_out_dims(input_dims::Tuple, cp::ConvParams)
    output_dims_conv = (((input_dims[1] - cp.filter_size[1] + 2 * cp.pad) / cp.stride_length) + 1)
    println(output_dims_conv)
    output_dims = ((output_dims_conv - cp.pool_window[1]) / cp.pool_stride) + 1
    return (Int(output_dims), Int(output_dims))
end

function num_params(cp::ConvParams)
    return prod([cp.filter_size..., cp.in_channels, cp.out_channels]) + cp.out_channels
end

function num_params(dp::DenseParams)
    return (dp.indim * dp.outdim) + dp.outdim
end

function layer_params(params_vec::AbstractVector, cp::ConvParams)
    bias = [pop!(params_vec) for _ in 1:cp.out_channels]
    weight = reshape(params_vec, cp.filter_size..., cp.in_channels, cp.out_channels)
    return weight, bias
end

function layer_params(params_vec::AbstractVector, dp::DenseParams)
    if dp.bias == true
        bias = [pop!(params_vec) for _ in 1:dp.outdim]
    else
        bias = [pop!(params_vec) for _ in 1:dp.outdim]
        # bias .*= 0.0
    end
    weight = reshape(params_vec, dp.outdim, dp.indim)
    return weight, bias
end

function split(x::AbstractVector, n)
    result = Vector{Vector{eltype(x)}}(undef, length(n))
    sum_elements = sum(n)
    if sum_elements == length(x)
        for i in 1:length(n)
            result[i] = splice!(x, 1:n[i])
        end
    end
    return result
end

function unpack_params(nn_params::AbstractVector, layers_spec::AbstractVector)
    num_params_list = num_params.(layers_spec)
    indices_list = split(nn_params, num_params_list)
    params_collection = [layer_params(i, j) for (i, j) in zip(indices_list, layers_spec)]
    return params_collection
end



###
### Conv Network specifications
###

# conv_layers = [ConvParams(), ConvParams(), ConvParams()]

# input_size = (128, 128)
# final_conv_out_dims = conv_out_dims(input_size, conv_layers[1]) |> x -> conv_out_dims(x, conv_layers[2]) |> x -> conv_out_dims(x, conv_layers[3])

# dense_layers = [DenseParams(indim = prod([final_conv_out_dims..., conv_layers[end].out_channels])), DenseParams(outdim = 10)]

# layers_spec = [conv_layers; dense_layers]

# function forward(x, nn_params::AbstractVector, layers_spec)
#     c1, c2, c3, d1, d2 = unpack_params(nn_params, layers_spec)
#     nn = Chain([layer(i..., j) for (i, j) in zip([c1, c2, c3], layers_spec[1:3])]..., Flux.flatten, [layer(i..., j) for (i, j) in zip([d1, d2], layers_spec[4:5])]..., softmax)
#     return nn(x)
# end

###
### Dense Network specifications
###

dense_layers = [DenseParams(indim = 150, outdim = 5, activation_fn = tanh), DenseParams(indim = 5, outdim = 3, activation_fn = tanh), DenseParams(indim = 3, outdim = 1, activation_fn = sigmoid)]

function forward(x, nn_params, layers_spec)
    nn = Chain([layer(i..., j) for (i, j) in zip(nn_params, layers_spec)]...)
    return nn(x)
end

### 
### Data specifications
### 

using DataFrames, DelimitedFiles

features = readdlm("Data/SECOM/nan_filtered_data.csv", ',', Float64)
# features = replace(features, NaN => 0)
labels = Int.(readdlm("Data/SECOM/nan_filtered_labels.csv")[:, 1])

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims = 1)) ./ std(x, dims = 1), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return (x .+ mean(orig, dims = 1)) .* std(orig, dims = 1)
end

features, _ = standardize(features)

using MultivariateStats

M = fit(PCA, features', maxoutdim = 150)
features_transformed = MultivariateStats.transform(M, features')

using Random
data = hcat(features_transformed', labels)
postive_data = data[data[:, end].==1.0, :]
negative_data = data[data[:, end].==-1.0, :]
data = vcat(postive_data, negative_data[1:100, :])
data = data[shuffle(axes(data, 1)), :]
# data = data[1:200, :]

# Function to split samples.
function split_data(df; at = 0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end

train, test = split_data(data, at = 0.9)

train_x = train[:, 1:end-1]
train_y = Int.(train[:, end])
train_y[train_y.==-1] .= 0
train_y = Bool.(train_y)
# train_y = hcat([Flux.onehot(i, [1, 2]) for i in train_y]...)
# train_data = Iterators.repeated((train_x', train_y_onehot), 128)

test_x = test[:, 1:end-1]
test_y = Int.(test[:, end])
test_y[test_y.==-1] .= 0
test_y = Bool.(test_y)
# test_y = hcat([Flux.onehot(i, [1, 2]) for i in test_y]...)

# total_params = sum(num_params.(dense_layers))
# d1, d2, d3 = unpack_params(randn(total_params), dense_layers)
# nn = Chain([layer(i..., j) for (i, j) in zip([d1, d2, d3], dense_layers)]...)
# forward(rand(376, 100), randn(total_params), dense_layers)

# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)


# Specify the probabilistic model.
@model function bayes_nn(x, y, layers_spec)
    total_num_params = sum(num_params.(layers_spec))

    # Create the weight and bias vector.
    nn_params ~ MvNormal(randn(total_num_params), sig .* ones(total_num_params))

    θ = unpack_params(nn_params, layers_spec)
    # Calculate predictions for the inputs given the weights
    # and biases in θ.
    ŷ = forward(x, θ, layers_spec)
    # println(size(ŷ))
    # Observe each prediction.
    for i = 1:length(y)
        # println(y[i], typeof(ŷ[i]))
        y[i] ~ Bernoulli(ŷ[i])
    end
end

###
### Perform Inference using VI
###

m = bayes_nn(train_x', train_y', dense_layers)
# q0 = Variational.meanfield(m)
advi = ADVI(10, 1000)
# opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(m, advi)


# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)

# params_samples = rand(q, 1000)
# params = mean.(eachrow(params_samples))
model = feedforward(params)
ŷ = model(test_x')
predictions = (ŷ .> 0.5)
# count(ŷ .> 0.7)
# count(test_y)

using MLJ
print("Accuracy:", accuracy(predictions, test_y'))
print("MCC:", mcc(predictions, test_y'))

# using Plots

# q_samples = rand(q, 10_000);

# p1 = histogram(q_samples[1, :], alpha = 0.7, label = "q");

# title!(raw"$\θ_1$")

# p2 = histogram(q_samples[2, :], alpha = 0.7, label = "q");

# title!(raw"$\θ_2$")

# plot(p1, p2)

###
### Perform Inference using MCMC
###

# chain = sample(bayes_nn(collect(features'), labels, dense_layers), NUTS(), 10)
# # Extract all weight and bias parameters.
# θ = MCMCChains.group(chain, :nn_params).value

#MCC metric to be used for imbalanced datasets

### 
### Prediction
### 

#we sample the params from the learned dist. q()
# then we feed the avaearged params to the netwrok and we perform predcitions on a test set