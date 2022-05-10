using DataFrames, DelimitedFiles

features = readdlm("Data/SECOM/secom_data.txt", ' ', Float64)
# features = replace(features, NaN => 0)
labels = Int.(readdlm("Data/SECOM/secom_labels.txt", ' ')[:, 1])

