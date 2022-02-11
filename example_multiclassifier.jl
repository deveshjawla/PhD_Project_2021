using Flux, RDatasets, Turing, Plots
gr()

iris = dataset("datasets", "iris")
using Random
iris = iris[shuffle(axes(iris, 1)), :]

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims = 1)) ./ std(x, dims = 1), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return (x .+ mean(orig, dims = 1)) .* std(orig, dims = 1)
end


iris_data, _ = standardize(Matrix(iris[:, 1:4]))
iris_labels = map(x -> x == "setosa" ? 1 : x == "versicolor" ? 2 : 3, iris[:, end])
iris = hcat(iris_data, iris_labels)
# Function to split samples.
function split_data(df; at = 0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end

train, test = split_data(iris, at = 0.9)

train_x = train[:, 1:4]
train_y = Int.(train[:, end])
train_y_onehot = hcat([Flux.onehot(i, [1, 2, 3]) for i in train_y]...)
train_data = Iterators.repeated((train_x', train_y_onehot), 128)

test_x = test[:, 1:4]
test_y = Int.(test[:, end])


# @df iris scatter(:SepalLength, :SepalWidth, group = :Species,
#     xlabel = "Length", ylabel = "Width", markersize = 5,
#     markeralpha = 0.75, markerstrokewidth = 0, linealpha = 0,
#     m = (0.5, [:cross :hex :star7], 12),
#     margin = 5mm)


function weights(theta::AbstractVector)
    W0 = reshape(theta[1:20], 5, 4)
    b0 = reshape(theta[21:25], 5)
    W1 = reshape(theta[26:40], 3, 5)
    b1 = reshape(theta[41:43], 3)
    return W0, b0, W1, b1
end

# nn = Chain(Dense(4, 5, relu), Dense(5, 3, relu), softmax)
# loss(x, y) = Flux.logitcrossentropy(nn(x), y)
# optimiser = Descent(0.01)
# Flux.train!(loss, Flux.params(nn), train_data, optimiser)

function feedforward(inp::AbstractArray, theta::AbstractVector)
    W0, b0, W1, b1 = weights(theta)
    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
        softmax
    )
    return model(inp)
end

alpha = 0.09;
sigma = sqrt(1.0 / alpha)

@model bayesnn(inp, lab) = begin
    theta ~ MvNormal(zeros(43), sigma .* ones(43))
    preds = feedforward(inp, theta)
    for i = 1:length(lab)
        # println(lab[i], Turing.ForwardDiff.value.(preds[:, i]))
        lab[i] ~ Categorical(preds[:, i])
    end
end

using ReverseDiff
Turing.setadbackend(:reversediff)
# chain = sample(bayesnn(Array(train_x'), train_y), NUTS(), 1000)
# theta = MCMCChains.group(chain, :theta).value

using Turing.Variational
m = bayesnn(Array(train_x'), train_y)
# q0 = Variational.meanfield(m) #Shall I use meanfield here? what other initial variational distribution?
advi = ADVI(10, 1000) #how many iteration? Any automatic convergence criteria?
# opt = Variational.DecayedADAGrad(0.1, 1.0, 0.9) #Schedule?
q = vi(m, advi)


# params_samples = rand(q, 1000)
# params = mean.(eachrow(params_samples))
# outputs = feedforward(test_x', params)

# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)

# using Plots

# q_samples = rand(q, 10000)

# p1 = histogram(q_samples[1, :], alpha = 0.7, label = "q")

# title!(raw"$\theta_1$")

# p2 = histogram(q_samples[2, :], alpha = 0.7, label = "q")

# title!(raw"$\theta_2$")

# plot(p1, p2)