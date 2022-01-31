using Flux, RDatasets, Turing, Plots
gr()

iris = dataset("datasets", "iris");

# @df iris scatter(:SepalLength, :SepalWidth, group = :Species,
#     xlabel = "Length", ylabel = "Width", markersize = 5,
#     markeralpha = 0.75, markerstrokewidth = 0, linealpha = 0,
#     m = (0.5, [:cross :hex :star7], 12),
#     margin = 5mm)

inputs = Matrix(iris[:, 1:4])
labels = map(x -> x == "setosa" ? 0 : x == "versicolor" ? 1 : 2, iris[:, end]);

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
        softmax
    )
    return model(inp)
end

alpha = 0.09;
sigma = sqrt(1.0 / alpha);

@model bayesnn(inp, lab) = begin
    theta ~ MvNormal(zeros(63), sigma .* ones(63))
    preds = feedforward(inp, theta)
    for i = 1:length(lab)
        # println(preds[:, i])
        lab[i] ~ Categorical(Turing.ForwardDiff.value.(preds[:, i]))
    end
end

# Turing.setadbackend(:reversediff)
# chain = sample(bayesnn(Array(inputs'), labels), NUTS(), 100)

using Turing.Variational
m = bayesnn(Array(inputs'), labels)
q0 = Variational.meanfield(m)
advi = ADVI(10, 100_000)
opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(m, advi, q0; optimizer = opt)
using AdvancedVI
AdvancedVI.elbo(advi, q, m, 1000)

using Plots

q_samples = rand(q, 100_000);

p1 = histogram(q_samples[1, :], alpha = 0.7, label = "q");

title!(raw"$\theta_1$")

p2 = histogram(q_samples[2, :], alpha = 0.7, label = "q");

title!(raw"$\theta_2$")

plot(p1, p2)