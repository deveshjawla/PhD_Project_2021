# # Function to split samples.
# function train_validate_test(df; v=0.6, t=0.8)
#     r = size(df, 1)
#     val_index = Int(round(r * v))
#     test_index = Int(round(r * t))
#     train = df[1:val_index, :]
#     validate = df[(val_index+1):test_index, :]
#     test = df[(test_index+1):end, :]
#     return train, validate, test
# end

### 
### Data
### 
PATH = @__DIR__
cd(PATH)
using DataFrames
using CSV
train_xy = CSV.read("./train.csv", DataFrame, header=1)
train_xy[train_xy.target.==-1, :target] .= 0
shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
train_xy = select(train_xy, vcat(shap_importances.feature_name[1:30], "target"))
using DelimitedFiles, Statistics, Random

using MLJ: partition

train_xy, validate_xy = partition(train_xy, 0.8, shuffle=true, rng=1334)

function data_balancing(data_xy; balancing::String)
    normal_data = data_xy[data_xy[:, end].==0.0, :]
    anomaly = data_xy[data_xy[:, end].==1.0, :]
    size_anomaly = size(anomaly)[1]
    size_normal = size(normal_data)[1]
    multiplier = div(size_normal, size_anomaly)
    leftover = mod(size_normal, size_anomaly)
    if balancing == "undersampling"
        data_xy = vcat(normal_data[1:size(anomaly)[1], :], anomaly)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "generative"
        new_anomaly = vcat(repeat(anomaly, outer=multiplier - 1), anomaly[1:leftover, :], anomaly)
        data_x = select(new_anomaly, Not([:target]))
        data_y = select(new_anomaly, [:target])
        new_anomaly = mapcols(x -> x + x * rand(collect(-0.05:0.01:0.05)), data_x)
        new_anomaly = hcat(data_x, data_y)
        data_xy = vcat(normal_data, new_anomaly)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "none"
        nothing
    end
    data_x = Matrix(data_xy)[:, 1:end-1]
    data_y = data_xy.target
    return data_x, data_y
end

train_x, train_y = data_balancing(train_xy, balancing="undersampling")
validate_x, validate_y = data_balancing(validate_xy, balancing="undersampling")


# A handy helper function to rescale our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ 0.000001)
end

train_mean = mean(train_x, dims=1)
train_std = std(train_x, dims=1)

train_x = standardize(train_x, train_mean, train_std)
validate_x = standardize(validate_x, train_mean, train_std)

# # # using MultivariateStats

# # # M = fit(PCA, train_x', maxoutdim=150)
# # # train_x_transformed = MultivariateStats.transform(M, train_x')

# # # # M = fit(PCA, test_x', maxoutdim = 150)
# # # test_x_transformed = MultivariateStats.transform(M, test_x')

# # # train_x = train_x_transformed'
# # # test_x = test_x_transformed'


name = "generative_calibration"

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
l1, l2, l3, l4, l5 = 100, 100, 20, 20, 1
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
ol5 = l4 * l5 + l5

total_num_params = nl1 + nl2 + nl3 + nl4 + ol5

using Flux

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:45400], 100, 454)
#     b0 = θ[45401:45500]
#     W1 = reshape(θ[45501:55500], 100, 100)
#     b1 = θ[55501:55600]
#     W2 = reshape(θ[55601:57600], 20, 100)
#     b2 = θ[57601:57620]
#     W3 = reshape(θ[57621:58020], 20, 20)
#     b3 = θ[58021:58040]
#     W4 = reshape(θ[58041:58060], 1, 20)
#     b4 = θ[58061:58061]
#     model = Chain(
#         Dense(W0, b0, leakyrelu),
#         Dense(W1, b1, leakyrelu),
#         Dense(W2, b2, leakyrelu),
#         Dense(W3, b3, leakyrelu),
#         Dense(W4, b4, sigmoid)
#     )
#     return model
# end

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


nn_initial = Chain(Dense(input_size, l1, relu), Dense(l1, l2, relu), Dense(l2, l3, relu), Dense(l3, l4, relu), Dense(l4, l5, σ))

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(nn_initial)

total_num_params = length(parameters_initial) # number of paraemters in NN


###
### Bayesian Network specifications
###

using Turing
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

###
### Inference
###

# ScikitLearn.CrossValidation.StratifiedKFold([ones(10)...,zeros(5)...], n_folds=5)

chain_timed = @timed sample(bayesnn(Array(train_x'), train_y, total_num_params, sigma, reconstruct), NUTS(), 10)
chain = chain_timed.value


summaries, quantiles = describe(chain);
sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
_, i = findmax(chain[:lp])
i = i.I[1]
θ[i, :]
elapsed = chain_timed.time
θ = MCMCChains.group(chain, :θ).value

# A helper to create NN from weights `theta` and run it through data `x`
nn_forward(x, θ) = reconstruct(θ)(x)

# Return the average predicted value across
# multiple weights.
function nn_predict(x, theta, num)
    return mean([nn_forward(x, theta[i, :])[1] for i in 1:10:num])
end;




params_set = collect.(eachrow(θ[:, :, 1]))

param_matrix = mapreduce(permutedims, vcat, params_set)
mkpath("./experiments/$(name)")
writedlm("./experiments/$(name)/param_matrix.csv", param_matrix, ',')


"""
Returns means, stds, classifications, majority_voting, majority_conf

where

means, stds are the average logits and std for all the chains

classifications are obtained using threshold

majority_voting, majority_conf are averaged classifications of each chain, if the majority i.e more than 0.5 chains voted in favor then we have a vote in favor of 1. the conf here means how many chains on average voted in favor of 1. majority vote is just rounding of majority conf.
"""
function predicitons_analyzer(test_xs, test_ys, params_set, threshold)::Tuple{Array{Float32},Array{Float32},Array{Int},Array{Int},Array{Float32}}
    means = []
    stds = []
    classifications = []
    majority_voting = []
    majority_conf = []
    for (test_x, test_y) in zip(eachrow(test_xs), test_ys)
        predictions = []
        for theta in params_set
            model = feedforward(theta)
            # make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predicitons to give us the final predictions_mean and std
            ŷ = model(collect(test_x))
            append!(predictions, ŷ)

        end
        individual_classifications = map(x -> ifelse(x > threshold, 1, 0), predictions)
        majority_vote = ifelse(mean(individual_classifications) > 0.5, 1, 0)
        majority_conf_ = mean(individual_classifications)
        ensemble_pred_prob = mean(predictions) #average logit
        std_pred_prob = std(predictions)
        ensemble_class = ensemble_pred_prob > threshold ? 1 : 0
        append!(means, ensemble_pred_prob)
        append!(stds, std_pred_prob)
        append!(classifications, ensemble_class)
        append!(majority_voting, majority_vote)
        append!(majority_conf, majority_conf_)
    end

    # for each samples mean, std in zip(means, stds)
    # plot(histogram, mean, std)
    # savefig(./plots of each sample)
    # end
    return means, stds, classifications, majority_voting, majority_conf
end

test_xy = CSV.read("./test.csv", DataFrame, header=1)
test_xy[test_xy.target.==-1, :target] .= 0
test_xy = select(test_xy, vcat(shap_importances.feature_name[1:30], "target"))
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

function conf_bin_indices(n, conf, test, predictions)
    bins = Dict{Int,Vector}()
    mean_conf = Dict{Int,Float32}()
    bin_acc = Dict{Int,Float32}()
    calibration_gaps = Dict{Int,Float32}()
    for i in 1:n
        lower = (i - 1) / n
        upper = i / n
        # println(lower, upper)
        bin = findall(x -> x > lower && x <= upper, conf)
        bins[i] = bin
        if length(predictions[bin]) > 1
            mean_conf_ = mean(conf[bin])
            mean_acc_ = count(==(1), test[bin]) / length(test[bin])
        else
            mean_conf_ = NaN
            mean_acc_ = NaN
        end
        println(length(predictions[bin]), ' ', mean_acc_)
        mean_conf[i] = mean_conf_
        bin_acc[i] = mean_acc_
        calibration_gaps[i] = abs(mean_acc_ - mean_conf_)
    end
    return bins, mean_conf, bin_acc, calibration_gaps
end


function ece_mce(bins, calibration_gaps, total_samples)
    n_bins = length(bins)
    ece_ = []
    for i in 1:n_bins
        append!(ece_, length(bins[i]) * calibration_gaps[i])
    end
    ece = sum(filter(!isnan, ece_)) / total_samples
    mce = maximum(filter(!isnan, collect(values(calibration_gaps))))
    return ece, mce
end
using Distributions
using Optim

# Logistic function for a scalar input:
function platt(conf::Float64)
    1.0 / (1.0 + exp(-conf))
end

function platt(conf)
    1.0 ./ (1.0 .+ exp.(-conf))
end

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