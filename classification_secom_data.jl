using Flux, RDatasets, Turing, Plots, DataFrames, DelimitedFiles
using ReverseDiff
Turing.setadbackend(:reversediff)
using Turing.Variational
gr()

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

using Random
data = hcat(features, labels)
data = data[shuffle(axes(data, 1)), :]

# Function to split samples.
function split_data(df; at = 0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end

train, test = split_data(data, at = 0.8)

train_x = train[:, 1:4]#end-1]
train_y = Int.(train[:, end])
train_y[train_y.==-1] .= 2
# train_y = Bool.(train_y)
# train_y = hcat([Flux.onehot(i, [1, 2]) for i in train_y]...)
# train_data = Iterators.repeated((train_x', train_y_onehot), 128)

test_x = test[:, 1:4]#end-1]
test_y = Int.(test[:, end])
test_y[test_y.==-1] .= 2
# test_y = Bool.(test_y)
# test_y = hcat([Flux.onehot(i, [1, 2]) for i in test_y]...)


function weights(theta::AbstractVector)
    W0 = reshape(theta[1:20], 5, 4)
    b0 = theta[21:25]
    W1 = reshape(theta[26:40], 3, 5)
    b1 = theta[41:43]
    W2 = reshape(theta[44:49], 2, 3)
    b2 = theta[50:51]
    return W0, b0, W1, b1, W2, b2
end

function feedforward(theta::AbstractVector)
    W0, b0, W1, b1, W2, b2 = weights(theta)
    model = Chain(
        Dense(W0, b0, tanh),
        Dense(W1, b1, tanh),
        Dense(W2, b2, relu),
        softmax
    )
    return model
end

alpha = 0.09
sigma = sqrt(1.0 / alpha)

@model bayesnn(inp, lab) = begin
    theta ~ MvNormal(zeros(51), sigma .* ones(51))
    model = feedforward(theta)
    preds = model(inp)
    for i = 1:length(lab)
        lab[i] ~ Categorical(preds[:,i])
    end
end

# chain = sample(bayesnn(Array(train_x'), train_y), NUTS(), 10)
# theta = MCMCChains.group(chain, :theta).value

m = bayesnn(train_x', train_y)
# q0 = Variational.meanfield(m) #Shall I use meanfield here? what other initial variational distribution?
advi = ADVI(10, 1000) #how many iteration? Any automatic convergence criteria?
# opt = Variational.DecayedADAGrad(0.1, 1.0, 0.9) #Schedule?
q = vi(m, advi)

# params_samples = rand(q, 1000)
# params = mean.(eachrow(params_samples))
# outputs = feedforward(test_x', params)