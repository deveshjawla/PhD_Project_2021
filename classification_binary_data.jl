PATH = @__DIR__
cd(PATH)

include("BNNUtils.jl")
include("Calibration.jl")
include("DataUtils.jl")

### 
### Data
### 
using DataFrames
using CSV
# train_xy = CSV.read("secom_data/train.csv", DataFrame, header=1)
train_xy = CSV.read("stroke_dataset/train.csv", DataFrame, header=1)
rename!(train_xy, :stroke => :target)
train_xy[train_xy.target.==-1, :target] .= 0
# shap_importances = CSV.read("secom_data/shap_importances.csv", DataFrame, header=1)
# train_xy = select(train_xy, vcat(shap_importances.feature_name[1:30], "target"))
shap_importances = CSV.read("stroke_dataset/shap_importances.csv", DataFrame, header=1)
train_xy = select(train_xy, vcat(shap_importances.feature_name[1:6], "target"))
using Statistics, Random

# using MLJ: partition

# train_xy, validate_xy = partition(train_xy, 0.8, shuffle=true, rng=1334)

train_x, train_y = data_balancing(train_xy, balancing="undersampling")
# validate_x, validate_y = data_balancing(validate_xy, balancing="undersampling")


# A handy helper function to rescale our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ 0.000001)
end

train_mean = mean(train_x, dims=1)
train_std = std(train_x, dims=1)

train_x = standardize(train_x, train_mean, train_std)
# validate_x = standardize(validate_x, train_mean, train_std)

# # # using MultivariateStats

# # # M = fit(PCA, train_x', maxoutdim=150)
# # # train_x_transformed = MultivariateStats.transform(M, train_x')

# # # # M = fit(PCA, test_x', maxoutdim = 150)
# # # test_x_transformed = MultivariateStats.transform(M, test_x')

# # # train_x = train_x_transformed'
# # # test_x = test_x_transformed'


name = "corrected"

mkpath("./experiments/$(name)")
# mkpath("./experiments/$(name)/DATA")
# writedlm("./experiments/$(name)/DATA/train_y.csv", train_y, ',')
# writedlm("./experiments/$(name)/DATA/train_x.csv", train_x, ',')
# writedlm("./experiments/$(name)/DATA/validate_y.csv", validate_y, ',')
# writedlm("./experiments/$(name)/DATA/validate_x.csv", validate_x, ',')
# writedlm("./experiments/$(name)/DATA/test_y.csv", test_y, ',')
# writedlm("./experiments/$(name)/DATA/test_x.csv", test_x, ',')

#reading back data
# name = "test_unbalanced"
# using DelimitedFiles
# train_y = vec(readdlm("./experiments/$(name)/DATA/train_y.csv", ',', Int))
# train_x = readdlm("./experiments/$(name)/DATA/train_x.csv", ',')
# validate_y = vec(readdlm("./experiments/$(name)/DATA/validate_y.csv", ',', Int))
# validate_x = readdlm("./experiments/$(name)/DATA/validate_x.csv", ',')
# using MLJ
# test_y = vec(readdlm("./experiments/$(name)/DATA/test_y.csv", ',', Int))
# test_x = readdlm("./experiments/$(name)/DATA/test_x.csv", ',')
# name = "test_unbalanced_relu"
# mkpath("./experiments/$(name)")


###
### Dense Network specifications
###

input_size = size(train_x)[2]
l1, l2, l3, l4, l5 = 10, 10, 5, 5 ,1
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
ol5 = l4 * l5 + l5

total_num_params = nl1 + nl2 + nl3 + nl4 + ol5

using Flux

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:3000], 100, 30)
#     b0 = θ[3001:3100]
#     W1 = reshape(θ[3101:13100], 100, 100)
#     b1 = θ[13101:13200]
#     W2 = reshape(θ[13201:15200], 20, 100)
#     b2 = θ[15201:15220]
#     W3 = reshape(θ[15221:15620], 20, 20)
#     b3 = θ[15621:15640]
#     W4 = reshape(θ[15641:15660], 1, 20)
#     b4 = θ[15661:15661]
#     model = Chain(
#         Dense(W0, b0, relu),
#         Dense(W1, b1, relu),
#         Dense(W2, b2, relu),
#         Dense(W3, b3, relu),
#         Dense(W4, b4, σ)
#     )
#     return model
# end


# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:600], 100, 6)
#     b0 = θ[601:700]
#     W1 = reshape(θ[701:10700], 100, 100)
#     b1 = θ[10701:10800]
#     W2 = reshape(θ[10801:12800], 20, 100)
#     b2 = θ[12801:12820]
#     W3 = reshape(θ[12821:13220], 20, 20)
#     b3 = θ[13221:13240]
#     W4 = reshape(θ[13241:13260], 1, 20)
#     b4 = θ[13261:13261]
#     model = Chain(
#         Dense(W0, b0, relu),
#         Dense(W1, b1, relu),
#         Dense(W2, b2, relu),
#         Dense(W3, b3, relu),
#         Dense(W4, b4, sigmoid)
#     )
#     return model
# end

nn_initial = Chain(Dense(input_size, l1, relu), Dense(l1, l2, relu), Dense(l2, l3, relu), Dense(l3, l4, relu), Dense(l4, l5, σ))

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(nn_initial)

total_num_params = length(parameters_initial) # number of paraemters in NN


###
### Bayesian Network specifications
###

using Turing
# setprogress!(false)
# using Zygote
# Turing.setadbackend(:zygote)
using ReverseDiff
Turing.setadbackend(:reversediff)

# alpha = 0.09
# sigma = sqrt(1.0 / alpha)

#Here we define the layer by layer initialisation
sigma = vcat(sqrt(2 / input_size) * ones(nl1), sqrt(2 / l1) * ones(nl2), sqrt(2 / l2) * ones(nl3), sqrt(2 / l3) * ones(nl4), sqrt(2 / l4) * ones(ol5))

@model bayesnn(x, y, nparameters, sigma, reconstruct) = begin
    θ ~ MvNormal(zeros(nparameters), sigma)
    nn = reconstruct(θ)
    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

# @model bayesnn(x, y) = begin
#     θ ~ MvNormal(zeros(total_num_params), sigma)
#     nn = feedforward(θ)
#     ŷ = nn(x)
#     for i = 1:lastindex(y)
#         y[i] ~ Bernoulli(ŷ[i])
#     end
# end

###
### Inference
###

# ScikitLearn.CrossValidation.StratifiedKFold([ones(10)...,zeros(5)...], n_folds=5)

chain_timed = @timed sample(bayesnn(Array(train_x'), train_y, total_num_params, sigma, reconstruct), NUTS(50, 0.65), 100)
# chain_timed = @timed sample(bayesnn(Array(train_x'), train_y), NUTS(50, 0.65), 1000)
chain = chain_timed.value

summaries, quantiles = describe(chain)
sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
_, i = findmax(chain[:lp])
i = i.I[1]
elapsed = chain_timed.time
θ = MCMCChains.group(chain, :θ).value
θ[i, :]

# # A helper to create NN from weights `theta` and run it through data `x`
# nn_forward(x, θ) = reconstruct(θ)(x)

# # Return the average predicted value across
# # multiple weights.
# function nn_predict(x, theta, start, step, stop)
#     return mean([nn_forward(x, theta[i, :])[1] for i in start:step:stop])
# end;




params_set = collect.(eachrow(θ[:, :, 1]))

param_matrix = mapreduce(permutedims, vcat, params_set)
mkpath("./experiments/$(name)")
writedlm("./experiments/$(name)/param_matrix.csv", param_matrix, ',')

###
### Perform Inference using VI
###

m = bayes_nn(train_x', train_y', dense_layers)
# q0 = Variational.meanfield(m)
advi = ADVI(10, 1000)
# opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(m, advi)


# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)

# params_samples = rand(q, 1000)
# params = mean.(eachrow(params_samples))
model = feedforward(params)
ŷ = model(test_x')
predictions = (ŷ .> 0.5)
# count(ŷ .> 0.7)
# count(test_y)

test_xy = CSV.read("stroke_dataset/test.csv", DataFrame, header=1)
rename!(test_xy, :stroke => :target)
test_xy[test_xy.target.==-1, :target] .= 0
test_xy = select(test_xy, vcat(shap_importances.feature_name[1:6], "target"))
test_x, test_y = data_balancing(test_xy, balancing="none")
test_x = standardize(test_x, train_mean, train_std)

threshold = 0.5

predictions_mean, predcitions_std, classifications, majority_vote, majority_conf = predicitons_analyzer(test_x, test_y, params_set, threshold)

writedlm("./experiments/$(name)/predcitions_std.csv", predcitions_std, ',')
writedlm("./experiments/$(name)/predictions_mean.csv", predictions_mean, ',')
writedlm("./experiments/$(name)/classifications.csv", classifications, ',')
writedlm("./experiments/$(name)/majority_vote.csv", majority_vote, ',')
writedlm("./experiments/$(name)/majority_conf.csv", majority_conf, ',')

# # Reading back results
# predcitions_std = vec(readdlm("./experiments/$(name)/predcitions_std.csv", ','))
# predictions_mean = vec(readdlm("./experiments/$(name)/predictions_mean.csv", ','))
# classifications = vec(readdlm("./experiments/$(name)/classifications.csv", ',', Int))
# majority_vote = vec(readdlm("./experiments/$(name)/majority_vote.csv", ',', Int))
# majority_conf = vec(readdlm("./experiments/$(name)/majority_conf.csv", ','))

ŷ = classifications
predicted_probs = predictions_mean

using EvalMetrics
using Flux: mse

f1 = f1_score(test_y, ŷ)
brier = mse(test_y, predicted_probs)
using Plots;
gr();
prplot(test_y, predicted_probs)
no_skill(x) = count(==(1), test_y) / length(test_y)
no_skill_score = no_skill(0)
plot!(no_skill, 0, 1, label="No Skill Classifier:$no_skill_score")
savefig("./experiments/$(name)/PRCurve.png")
# mcc = matthews_correlation_coefficient(test_y, ŷ)
# acc = accuracy(ŷ, test_y)
fpr = false_positive_rate(test_y, ŷ)
# fnr = fnr(ŷ, test_y)
# tpr = tpr(ŷ, test_y)
# tnr = tnr(ŷ, test_y)
prec = precision(test_y, ŷ)
# recall = true_positive_rate(ŷ, test_y)
prauc = au_prcurve(test_y, predicted_probs[:, 1])

writedlm("./experiments/$(name)/results.txt", [["elapsed", "threshold", "brier", "f1", "fpr", "precision", "PRAUC"] [elapsed, threshold, brier, f1, fpr, prec, prauc]], ',')

####
#### Calibration
####

number_of_bins = 3

using Distributions
using Optim

predictions_mean_validate, predcitions_std_validate, classifications_validate, majority_vote_validate, majority_conf_validate = predicitons_analyzer(validate_x, validate_y, params_set, threshold)

labels = validate_y
ŷ_validate = classifications_validate
# labels[labels.==2] .= 0
# ŷ_validate[ŷ_validate.==2] .= 0
pred_conf = predictions_mean_validate

loss((a, b)) = -sum(labels[i] * log(platt(pred_conf[i] * a + b)) + (1.0 - labels[i]) * log(1.0 - platt(pred_conf[i] * a + b)) for i = 1:lastindex(pred_conf))

@time result = optimize(loss, [1.0, 1.0], LBFGS())

a, b = result.minimizer

#calibrating the results of test set

calibrated_pred_prob = platt(predictions_mean .* a .+ b)
classifications = round.(Int, calibrated_pred_prob)

# Calibrated Results

bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, calibrated_pred_prob, test_y, ŷ)

total_samples = lastindex(calibrated_pred_prob)

ECE, MCE = ece_mce(bins, calibration_gaps, total_samples)

writedlm("./experiments/$(name)/validate_ece_mce.txt", [ECE, MCE])

f(x) = x
using Plots
reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false, title="Reliability diagram with \n ECE:$(ECE), MCE:$(MCE)",
    xlabel="Confidence",
    ylabel="# Class labels in Target", size=(800, 600))
plot!(f, 0, 1, label="Perfect Calibration")
savefig(reliability_diagram, "./experiments/$(name)/reliability_diagram.png")



# using Distributions
# using Optim

# # Logistic function for a scalar input:
# function platt(conf::Float64)
#     1.0 / (1.0 + exp(-conf))
# end

# function platt(conf)
#     1.0 ./ (1.0 .+ exp.(-conf))
# end

# labels = validate_y
# predictions_mean, predcitions_std, classifications, majority_vote, majority_conf = predicitons_analyzer(validate_x, validate_y, params_set)
# pred_conf = predictions_mean

# loss((a, b)) = -sum(labels[i] * log(platt(pred_conf[i] * a + b)) + (1.0 - labels[i]) * log(1.0 - platt(pred_conf[i] * a + b)) for i = 1:lastindex(pred_conf))

# @time result = optimize(loss, [1.0, 1.0], LBFGS())

# a, b = result.minimizer


# #calibrating the results of test set

# number_of_bins = 10

# predictions_mean, predcitions_std, classifications, majority_vote, majority_conf = predicitons_analyzer(test_x, test_y, params_set)

# bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, predictions_mean, test_y, classifications)

# reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false)
# savefig(reliability_diagram, "./experiments/$(name)/test_reliability_diagram.png")

# total_samples = lastindex(predictions_mean)

# ECE = ece(bins, calibration_gaps, total_samples)
# MCE = mce(calibration_gaps)

# writedlm("./experiments/$(name)/test_ece_mce.txt", [ECE, MCE])

# calibrated_pred_prob = platt(predictions_mean .* -a .- b)
# classifications = round.(Int, calibrated_pred_prob)

# bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, calibrated_pred_prob, test_y, classifications)

# reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false)
# savefig(reliability_diagram, "./experiments/$(name)/calibrated_test_reliability_diagram.png")

# total_samples = lastindex(calibrated_pred_prob)

# ECE = ece(bins, calibration_gaps, total_samples)
# MCE = mce(calibration_gaps)

# writedlm("./experiments/$(name)/calibrated_test_ece_mce.txt", [ECE, MCE])