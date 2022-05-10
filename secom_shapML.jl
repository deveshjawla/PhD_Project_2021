### 
### Data
### 

using DataFrames, DelimitedFiles, Statistics

# features = readdlm("Data/SECOM/nan_filtered_data.csv", ',', Float64)
# # features = replace(features, NaN => 0)
# labels = Int.(readdlm("Data/SECOM/nan_filtered_labels.csv")[:, 1])

features = readdlm("Data/SECOM/secom_data_preprocessed_moldovan2017.csv", ',', Float32)

labels = Int.(readdlm("Data/SECOM/secom_labels.txt", ' ')[:, 1])


# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- minimum(x, dims=1)) ./ (maximum(x, dims =1) - minimum(x, dims =1) .+ 0.000001)
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

train_x = standardize(train_x)
test_x = standardize(test_x)

train = hcat(train_x, train_y)

postive_data = train[train[:, end].==1.0, :]
negative_data = train[train[:, end].==0.0, :]
train = vcat(postive_data, negative_data[1:100, :])
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
    W0 = reshape(θ[1:4086], 9, 454)
    b0 = θ[4087:4095]
    W1 = reshape(θ[4096:4122], 3, 9)
    b1 = θ[4123:4125]
    W2 = reshape(θ[4126:4128], 1, 3)
    b2 = θ[4129:4129]

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
    θ ~ MvNormal(zeros(4129), sigma .* ones(4129))
    nn = feedforward(θ)
    ŷ = nn(x)
    for i = 1:length(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

###
### Inference
###

chain = sample(bayesnn(Array(train_x'), train_y), NUTS(), 100)
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

# using ShapML
# using MLJ  # Machine learning
# using Gadfly  # Plotting

# # Create a wrapper function that takes the following positional arguments: (1) a
# # trained ML model from any Julia package, (2) a DataFrame of model features. The
# # function should return a 1-column DataFrame of predictions--column names do not matter.
# function predict_function(model, data)
#     data_pred = DataFrame(model(Matrix(data)')', :auto)
#     return data_pred
# end


# sample_size = 60  # Number of Monte Carlo samples.
# #------------------------------------------------------------------------------
# # Compute stochastic Shapley values.
# data_shap = ShapML.shap(explain=DataFrame(train_x, :auto),
#     reference=copy(DataFrame(train_x, :auto)),
#     model=model,
#     predict_function=predict_function,
#     sample_size=sample_size,
#     seed=1
# )

# show(data_shap, allcols=true)

# gd = groupby(data_shap, :feature_name)
# data_plot = combine(gd, :shap_effect => x-> mean(abs.(x)))

# data_plot = sort(data_plot, order(:shap_effect_function, rev=true))

# baseline = round(data_shap.intercept[1], digits=1)

# p = plot(data_plot[1:30,:], y=:feature_name, x=:shap_effect_function, Coord.cartesian(yflip=true),
#     Scale.y_discrete, Geom.bar(position=:dodge, orientation=:horizontal),
#     Theme(bar_spacing=1mm),
#     Guide.xlabel("|Shapley effect| (baseline = $baseline)"), Guide.ylabel(nothing),
#     Guide.title("Feature Importance - Mean Absolute Shapley Value"))

ŷ = model(test_x')
predictions = (ŷ .> 0.5)
# count(ŷ .> 0.7)
# count(test_y)

using MLJ
print("Accuracy:", accuracy(predictions, test_y'))
print("MCC:", mcc(predictions, test_y'))
print("F1:", f1score(predictions, test_y'))
print("TPR:", tpr(predictions, test_y'))
print("FPR:", fpr(predictions, test_y'))
print("FNR:", fnr(predictions, test_y'))
print("TNR:", tnr(predictions, test_y'))


# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)