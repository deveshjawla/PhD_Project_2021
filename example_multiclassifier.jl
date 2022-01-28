using Flux, RDatasets, Measures, StatsPlots, PlotThemes, Turing, ReverseDiff
gr()

iris = dataset("datasets", "iris");

@df iris scatter(:SepalLength, :SepalWidth, group = :Species,
    xlabel = "Length", ylabel = "Width", markersize = 5,
    markeralpha = 0.75, markerstrokewidth = 0, linealpha = 0,
    m = (0.5, [:cross :hex :star7], 12),
    margin = 5mm)

inputs = Matrix(iris[:, 1:4])
labels = map(x -> x == "setosa" ? 0 : x == "versicolor" ? 1 : 2, iris[:, end]);

function softmax_(arr::AbstractArray)
    ex = mapslices(x -> exp.(x), arr, dims = 1)
    rows, cols = size(arr)
    val = similar(ex)
    for i in 1:cols
        s = sum(ex[:, i])
        for j in 1:rows
            val[j, i] = ex[j, i] / s
        end
    end
    return val
end

function weights(theta::AbstractVector)
    W0 = reshape(theta[1:20], 5, 4)
    b0 = reshape(theta[21:25], 5)
    W1 = reshape(theta[26:45], 4, 5)
    b1 = reshape(theta[46:49], 4)
    WO = reshape(theta[50:61], 3, 4)
    bO = reshape(theta[61:63], 3)
    return W0, b0, W1, b1, WO, bO
end

function feedforward(inp::AbstractArray, theta::AbstractVector)
    W0, b0, W1, b1, W2, b2 = weights(theta)
    model = Chain(
        Dense(W0, b0, tanh),
        Dense(W1, b1, tanh),
        Dense(W2, b2, σ),
        softmax_
    )
    return model(inp)
end

alpha = 0.09;
sigma = sqrt(1.0 / alpha);

@model bayesnn(inp, lab) = begin
    theta ~ MvNormal(zeros(63), sigma .* ones(63))

    preds = feedforward(inp, theta)
    for i = 1:length(lab)
        lab[i] ~ Categorical(preds[:, i])
    end
end

Turing.setadbackend(:reversediff)
chain = sample(bayesnn(Array(inputs'), labels), NUTS(200, 0.65), 100)