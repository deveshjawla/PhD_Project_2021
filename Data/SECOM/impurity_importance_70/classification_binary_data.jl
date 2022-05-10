
# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- minimum(x, dims=1)) ./ (maximum(x, dims=1) - minimum(x, dims=1) .+ 0.000001)
end

# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end


### 
### Data
### 
PATH = @__DIR__
using DataFrames, DelimitedFiles, Statistics
features = readdlm(PATH*"/seventy_imp_features.csv", ',', Float64)
labels = Int.(readdlm(PATH*"/secom_labels.txt")[:, 1])

using Random
data = hcat(features, labels)
data = data[shuffle(axes(data, 1)), :]
train, test = split_data(data, at=0.8)

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

train_x = standardize(train_x)
test_x = standardize(test_x)

# using MultivariateStats

# M = fit(PCA, train_x', maxoutdim=150)
# train_x_transformed = MultivariateStats.transform(M, train_x')

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, test_x')

# train_x = train_x_transformed'
# test_x = test_x_transformed'

train = hcat(train_x, train_y)

postive_data = train[train[:, end].==1.0, :]
negative_data = train[train[:, end].==0.0, :]
train = vcat(postive_data, negative_data[1:size(postive_data)[1], :])
# data = data[1:200, :]
train = train[shuffle(axes(train, 1)), :]


train_x = train[:, 1:end-1]
train_y = Int.(train[:, end])
train_y[train_y.==-1] .= 0
train_y = Bool.(train_y)

###
### Dense Network specifications
###

using Flux

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:630], 9, 70)
    b0 = θ[631:639]
    W1 = reshape(θ[640:666], 3, 9)
    b1 = θ[667:669]
    W2 = reshape(θ[669:671], 1, 3)
    b2 = θ[672:672]

    model = Chain(
        Dense(W0, b0, tanh),
        Dense(W1, b1, tanh),
        Dense(W2, b2, sigmoid)
    )
    return model
end
###
### Bayesian Network specifications
###

using ReverseDiff, Turing
Turing.setadbackend(:reversediff)

alpha = 0.09
sigma = sqrt(1.0 / alpha)

@model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(672), sigma .* ones(672))
    nn = feedforward(θ)
    ŷ = nn(x)
    for i = 1:length(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

###
### Inference
###

chain_timed = @timed sample(bayesnn(Array(train_x'), train_y), NUTS(), 100)
chain = chain_timed.value
elapsed = chain_timed.time
θ = MCMCChains.group(chain, :θ).value
params = mean.(eachcol(θ[:, :, 1]))
params_std = std.(eachcol(θ[:, :, 1]))

# using Turing.Variational

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
mcc = MLJ.mcc(predictions, test_y')
f1 = MLJ.f1score(predictions, test_y')
acc = MLJ.accuracy(predictions, test_y')
fpr = MLJ.fpr(predictions, test_y')
fnr = MLJ.fnr(predictions, test_y')
tpr = MLJ.tpr(predictions, test_y')
tnr = MLJ.tnr(predictions, test_y')
prec = MLJ.precision(predictions, test_y')
recall = MLJ.recall(predictions, test_y')

writedlm(PATH*"/seventy_imp_features.txt", [elapsed, mcc, f1, acc, fpr, fnr, tpr, tnr, prec, recall],',')


# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)