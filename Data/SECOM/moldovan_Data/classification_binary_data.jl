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
features = readdlm(PATH * "/secom_data_preprocessed_moldovan2017.csv", ',', Float64)
labels = Int.(readdlm(PATH * "/secom_labels.txt")[:, 1])

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

# A handy helper function to rescale our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ 0.000001)
end

train_mean = mean(train_x, dims=1)
train_std = std(train_x, dims=1)

train_x = standardize(train_x, train_mean, train_std)
test_x = standardize(test_x, train_mean, train_std)


using MultivariateStats

M = fit(PCA, train_x', maxoutdim=150)
train_x_transformed = MultivariateStats.transform(M, train_x')

# M = fit(PCA, test_x', maxoutdim = 150)
test_x_transformed = MultivariateStats.transform(M, test_x')

train_x = train_x_transformed'
test_x = test_x_transformed'

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

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:4086], 9, 454)
#     b0 = θ[4087:4095]
#     W1 = reshape(θ[4096:4122], 3, 9)
#     b1 = θ[4123:4125]
#     W2 = reshape(θ[4126:4128], 1, 3)
#     b2 = θ[4129:4129]

#     model = Chain(
#         Dense(W0, b0, tanh),
#         Dense(W1, b1, tanh),
#         Dense(W2, b2, sigmoid)
#     )
#     return model
# end

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:1350], 9, 150)
    b0 = θ[1351:1359]
    W1 = reshape(θ[1360:1386], 3, 9)
    b1 = θ[1387:1389]
    W2 = reshape(θ[1390:1392], 1, 3)
    b2 = θ[1393:1393]
    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
        Dense(W2, b2, sigmoid)
    )
    return model
end
###
### Bayesian Network specifications
###

using Turing
# using Zygote
# Turing.setadbackend(:zygote)
using ReverseDiff
Turing.setadbackend(:reversediff)

alpha = 0.09
sigma = sqrt(1.0 / alpha)

# @model bayesnn(x, y) = begin
#     θ ~ MvNormal(zeros(4192), sigma .* ones(4192))
#     nn = feedforward(θ)
#     ŷ = nn(x)
#     for i = 1:length(y)
#         y[i] ~ Bernoulli(ŷ[i])
#     end
# end

@model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(1393), sigma .* ones(1393))
    nn = feedforward(θ)
    ŷ = nn(x)
    for i = 1:length(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

###
### Inference
###
δ = "relu"

chain_timed = @timed sample(bayesnn(Array(train_x'), train_y), NUTS(), 1000)
chain = chain_timed.value
elapsed = chain_timed.time
θ = MCMCChains.group(chain, :θ).value

params_set = collect.(eachrow(θ[:, :, 1]))

function predicitons_analyzer(test_xs, test_ys, params_set)
    means = []
    stds = []
    accuracies = []
    for (test_x, test_y) in zip(eachrow(test_xs), test_ys)
        predictions = []
        for theta in params_set
            model = feedforward(theta)
            ŷ = model(collect(test_x))
            append!(predictions, ŷ)
        end
        mean_precciton = mean(predictions)
        std_predciton = std(predictions)
        accuracy = round(Int, mean_precciton) == test_y ? 1 : 0
        append!(means, mean_precciton)
        append!(stds, std_predciton)
        append!(accuracies, accuracy)
    end

    # for each samples mean, std in zip(means, stds)
    # plot(histogram, mean, std)
    # savefig(./plots of each sample)
    # end
    return means, stds, accuracies
end

predictions_mean, predcitions_std, predictions_acc = predicitons_analyzer(test_x, test_y, params_set)


param_matrix = mapreduce(permutedims, vcat, params_set)
mkdir("./test_results_$(δ)")
writedlm("./test_results_$(δ)/train_y.csv", train_y, ",")
writedlm("./test_results_$(δ)/train_x.csv", train_x, ",")
writedlm("./test_results_$(δ)/test_y.csv", test_y, ",")
writedlm("./test_results_$(δ)/test_x.csv", test_x, ",")
writedlm("./test_results_$(δ)/param_matrix.csv", param_matrix, ",")
writedlm("./test_results_$(δ)/predcitions_std.csv", predcitions_std, ",")
writedlm("./test_results_$(δ)/predictions_mean.csv", predictions_mean, ",")
writedlm("./test_results_$(δ)/predictions_acc.csv", predictions_acc, ",")

params = mean.(eachcol(θ[:, :, 1]))
params_std = std.(eachcol(θ[:, :, 1]))

# using Turing.Variational

# m = bayesnn(train_x', train_y')
# # q0 = Variational.meanfield(m) #Shall I use meanfield here? what other initial variational distribution?
# advi = ADVI(10, 1000) #how many iteration? Any automatic convergence criteria?
# # opt = Variational.DecayedADAGrad(0.1, 1.0, $(δ)) #Schedule?
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

writedlm("./test_results_$(δ)/secom_data_preprocessed_moldovan2017_PCA.txt", [elapsed, mcc, f1, acc, fpr, fnr, tpr, tnr, prec, recall], ',')


# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)