# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end

# A handy helper function to rescale our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ 0.000001)
end

### 
### Data
###
PATH = @__DIR__
using Flux, Turing
using CSV
using DataFrames

iris = CSV.read("Data/IRIS/Iris_cleaned.csv", DataFrame, header=1)
target = "Species"

using Random
iris = iris[shuffle(axes(iris, 1)), :]
train, test = split_data(iris, at=0.8)

train_mean = mean(Matrix(train[:, 1:4]), dims=1)
train_std = std(Matrix(train[:, 1:4]), dims=1)

train_x = standardize(Matrix(train[:, 1:4]), train_mean, train_std)
train_y = train[:, end]

test_x = standardize(Matrix(test[:, 1:4]), train_mean, train_std)
test_y = test[:, end]

train_y_onehot = hcat([Flux.onehot(i, [1, 2, 3]) for i in train_y]...)

test_y_onehot = hcat([Flux.onehot(i, [1, 2, 3]) for i in test_y]...)

###
### Plotting
###
# @df iris scatter(:SepalLength, :SepalWidth, group = :Species,
#     xyel = "Length", yyel = "Width", markersize = 5,
#     markeralpha = 0.75, markerstrokewidth = 0, linealpha = 0,
#     m = (0.5, [:cross :hex :star7], 12),
#     margin = 5mm)

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:20], 5, 4)
    b0 = reshape(θ[21:25], 5)
    W1 = reshape(θ[26:40], 3, 5)
    b1 = reshape(θ[41:43], 3)
    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
        softmax
    )
    return model
end

using ReverseDiff
Turing.setadbackend(:reversediff)

alpha = 0.09
sigma = sqrt(1.0 / alpha)

activations_file = open("./activations_file.txt")
weights_file = open("./weights_file.txt")
activations_file = open("./activations_file.txt")

@model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(43), sigma .* ones(43))
    nn = feedforward(θ)

    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(ŷ[:, i])
    end
end

close(activations_file)
close(weights_file)

###
### Inference
###
name = "iris_0.8_split"

chain_timed = @timed sample(bayesnn(Array(train_x'), train_y), NUTS(50, 0.65), 100)
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
            append!(predictions, argmax(ŷ))
        end
        mean_prediction = mean(predictions)
        std_predciton = std(predictions)
        accuracy = round(Int, mean_prediction) == test_y ? 1 : 0
        append!(means, mean_prediction)
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
mkdir("./test_results_$(name)")
writedlm("./test_results_$(name)/train_y.csv", train_y, ",")
writedlm("./test_results_$(name)/train_x.csv", train_x, ",")
writedlm("./test_results_$(name)/test_y.csv", test_y, ",")
writedlm("./test_results_$(name)/test_x.csv", test_x, ",")
writedlm("./test_results_$(name)/param_matrix.csv", param_matrix, ",")
writedlm("./test_results_$(name)/predcitions_std.csv", predcitions_std, ",")
writedlm("./test_results_$(name)/predictions_mean.csv", predictions_mean, ",")
writedlm("./test_results_$(name)/predictions_acc.csv", predictions_acc, ",")

params_ = mean.(eachcol(θ[:, :, 1]))
params_std = std.(eachcol(θ[:, :, 1]))

model = feedforward(params_)
ŷ = model(test_x')
predictions = argmax.(eachcol(ŷ))

using MLJ
acc = MLJ.accuracy(predictions, test_y)
acc_ = MLJ.accuracy(round.(Int, predictions_mean), test_y)

CSV.write("results.txt", [elapsed, acc, acc_], ',')