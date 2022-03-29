### 
### Data
### 

using DataFrames, DelimitedFiles, Statistics

features = readdlm("Data/SECOM/nan_filtered_data.csv", ',', Float64)
# features = replace(features, NaN => 0)
labels = Int.(readdlm("Data/SECOM/nan_filtered_labels.csv")[:, 1])

sum([i>0.5 for i in labels])
sum([i<0.5 for i in labels])

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims = 1)) ./ (std(x, dims = 1) .+ 0.000001), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return (x .+ mean(orig, dims = 1)) .* std(orig, dims = 1)
end

# Function to split samples.
function split_data(df; at = 0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end


using Random, MLJ
data = hcat(features, labels)
data = data[shuffle(axes(data, 1)), :]
train, test = split_data(data, at = 0.7)


sum([i>0.5 for i in train[:,end]])
sum([i<0.5 for i in train[:,end]])

sum([i>0.5 for i in test[:,end]])
sum([i<0.5 for i in test[:,end]])


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

# train_x, _ = standardize(train_x)
# test_x, _ = standardize(test_x)

# using MultivariateStats

# M = fit(PCA, train_x', maxoutdim = 150)
# train_x_transformed = MultivariateStats.transform(M, train_x')

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, test_x')

# train_x = Matrix(train_x_transformed')
# test_x = Matrix(test_x_transformed')

train = hcat(train_x, train_y)

postive_data = train[train[:, end].==1.0, :]
negative_data = train[train[:, end].==0.0, :]
train = vcat(postive_data, negative_data[1:88, :])
data = data[1:200, :]
train = train[shuffle(axes(train, 1)), :]


train_x = train[:, 1:end-1]
train_y = Int.(train[:, end])
train_y[train_y .== -1] .= 0
train_y = Bool.(train_y)

###
### Dense Network specifications
###

sum([i>0.5 for i in train_y])
sum([i<0.5 for i in train_y])

sum([i>0.5 for i in test_y])
sum([i<0.5 for i in test_y])



using MLJ
import XGBoost
import MLJXGBoostInterface

XGB = @load XGBoostClassifier verbosity=0 

# train_y=coerce(DataFrame(real=train_y),:real=>Finite)
# test_y=coerce(DataFrame(real=test_y),:real=>OrderedFactor)
train_y, _ =unpack(coerce(DataFrame(real=train_y),
            :real=>OrderedFactor),
        ==(:real),
        colname -> true, );

test_y, _ =unpack(coerce(DataFrame(real=test_y),
        :real=>OrderedFactor),
    ==(:real),
    colname -> true, );

machbest = machine(
    #     self_tune,
    #     XGB(num_round=30, booster="dart", eta = 1.0, max_depth = 7, rate_drop=0.1),
        XGB(nthread=6, num_round=50, booster="gbtree", eta = 1.0, max_depth = 8,
            scale_pos_weight=13,
            min_child_weight=1,
            ),
            DataFrame(train_x,:auto),train_y
    )

fit!(machbest,verbosity=1 )


sum([i==true for i in train_y])
length(train_y)

predictions = vec(pdf(MLJ.predict(machbest, DataFrame(test_x,:auto) ),[true]))

test_y

print("Accuracy:", accuracy([i>0.05 for i in predictions], [i==true for i in test_y]))
print("MCC:", mcc([i>0.005 for i in predictions], [i==true for i in test_y]))

ConfusionMatrix()([i>0.05 for i in predictions], [i==true for i in test_y])

