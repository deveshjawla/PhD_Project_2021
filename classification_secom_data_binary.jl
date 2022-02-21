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

function weights(θ::AbstractVector)
    W0 = reshape(θ[1:1350], 9, 150)
    b0 = θ[1351:1359]
    W1 = reshape(θ[1360:1386], 3, 9)
    b1 = θ[1387:1389]
    W2 = reshape(θ[1390:1392], 1, 3)
    b2 = θ[1393:1393]
    return W0, b0, W1, b1, W2, b2
end

# function weights(θ::AbstractVector)
#     W0 = reshape(θ[1:1500], 10, 150)
#     b0 = θ[1501:1510]
#     W1 = reshape(θ[1511:1560], 5, 10)
#     b1 = θ[1561:1565]
#     W2 = reshape(θ[1566:1570], 1, 5)
#     b2 = θ[1571:1571]
#     return W0, b0, W1, b1, W2, b2
# end

function feedforward(θ::AbstractVector)
    W0, b0, W1, b1, W2, b2 = weights(θ)
    model = Chain(
        Dense(W0, b0, tanh),
        Dense(W1, b1, tanh),
        Dense(W2, b2, sigmoid)
    )
    return model
end

alpha = 0.09
sigma = sqrt(1.0 / alpha)

@model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(1393), sigma .* ones(1393))
    nn = feedforward(θ)
    ŷ = nn(x)
    for i = 1:length(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

chain = sample(bayesnn(Array(train_x'), train_y), NUTS(), 1000)
θ = MCMCChains.group(chain, :θ).value
params = mean.(eachcol(θ[:,:,1]))

# m = bayesnn(train_x', train_y')
# # q0 = Variational.meanfield(m) #Shall I use meanfield here? what other initial variational distribution?
# advi = ADVI(10, 1000) #how many iteration? Any automatic convergence criteria?
# # opt = Variational.DecayedADAGrad(0.1, 1.0, 0.9) #Schedule?
# q = vi(m, advi)

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


# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)