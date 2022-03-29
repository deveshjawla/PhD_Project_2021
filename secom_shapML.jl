### 
### Data
### 

using DataFrames, DelimitedFiles, Statistics

features = readdlm("Data/SECOM/nan_filtered_data.csv", ',', Float64)
# features = replace(features, NaN => 0)
labels = Int.(readdlm("Data/SECOM/nan_filtered_labels.csv")[:, 1])

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims=1)) ./ (std(x, dims=1) .+ 0.000001), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return (x .+ mean(orig, dims=1)) .* std(orig, dims=1)
end

# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end


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

train_x, _ = standardize(train_x)
test_x, _ = standardize(test_x)

train = hcat(train_x, train_y)

postive_data = train[train[:, end].==1.0, :]
negative_data = train[train[:, end].==0.0, :]
train = vcat(postive_data, negative_data[1:88, :])
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
    W0 = reshape(θ[1:3384], 9, 376)
    b0 = θ[3385:3393]
    W1 = reshape(θ[3394:3420], 3, 9)
    b1 = θ[3421:3423]
    W2 = reshape(θ[3424:3426], 1, 3)
    b2 = θ[3427:3427]

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
    θ ~ MvNormal(zeros(3427), sigma .* ones(3427))
    nn = feedforward(θ)
    ŷ = nn(x)
    for i = 1:length(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

###
### Inference
###

chain = sample(bayesnn(Array(train_x'), train_y), NUTS(), 500)
θ = MCMCChains.group(chain, :θ).value
params = mean.(eachcol(θ[:, :, 1]))

# using Turing.Variational

# m = bayesnn(train_x', train_y')
# # q0 = Variational.meanfield(m) #Shall I use meanfield here? what other initial variational distribution?
# advi = ADVI(10, 1000) #how many iteration? Any automatic convergence criteria?
# # opt = Variational.DecayedADAGrad(0.1, 1.0, 0.9) #Schedule?
# q = vi(m, advi)

# params_samples = rand(q, 1000)
# params = mean.(eachrow(params_samples))
model = feedforward(params)

function model_(model, data)

end

using ShapML
using MLJ  # Machine learning
using Gadfly  # Plotting

# Create a wrapper function that takes the following positional arguments: (1) a
# trained ML model from any Julia package, (2) a DataFrame of model features. The
# function should return a 1-column DataFrame of predictions--column names do not matter.
function predict_function(model, data)
    data_pred = DataFrame(model(Matrix(data)')', :auto)
    return data_pred
end


sample_size = 60  # Number of Monte Carlo samples.
#------------------------------------------------------------------------------
# Compute stochastic Shapley values.
data_shap = ShapML.shap(explain=DataFrame(train_x, :auto),
    reference=copy(DataFrame(train_x, :auto)),
    model=model,
    predict_function=predict_function,
    sample_size=sample_size,
    seed=1
)

show(data_shap, allcols=true)

gd = groupby(data_shap, :feature_name)
data_plot = combine(gd, :shap_effect => x-> mean(abs.(x)))

data_plot = sort(data_plot, order(:shap_effect_function, rev=true))

baseline = round(data_shap.intercept[1], digits=1)

p = plot(data_plot[1:30,:], y=:feature_name, x=:shap_effect_function, Coord.cartesian(yflip=true),
    Scale.y_discrete, Geom.bar(position=:dodge, orientation=:horizontal),
    Theme(bar_spacing=1mm),
    Guide.xlabel("|Shapley effect| (baseline = $baseline)"), Guide.ylabel(nothing),
    Guide.title("Feature Importance - Mean Absolute Shapley Value"))

ŷ = model(test_x')
predictions = (ŷ .> 0.5)
# count(ŷ .> 0.7)
# count(test_y)

using MLJ
print("Accuracy:", accuracy(predictions, test_y'))
print("MCC:", mcc(predictions, test_y'))


# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)