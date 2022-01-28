using CSV, DataFrames, DelimitedFiles

features = readdlm("Data/secom_data.txt")
labels = Int.(readdlm("Data/secom_labels.txt")[:,1])



