using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs, crossentropy #Note on crossentropy,
#when using softmax layer use crossentropy otherwise use logitcrossentropy
using Base.Iterators: repeated
using Parameters: @with_kw
using Printf, BSON
using Random
using DelimitedFiles
using CSV
using DataFrames
using LinearAlgebra:norm
using CUDA
if has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    η::Float64 = 0.1
    epochs::Int = 100
    batchsize::Int = 64
    savepath::String = "/home/jawla/Simulations/"
    device::Function = gpu
	lattice_size::Int = 128
	sample_size::Int = 1602
	n_features::Int = 4224
	test_sample_size::Int = 1351
end


function getdata(args)
    # Loading Dataset
    df=Array{Float32}(undef,args.sample_size,args.n_features+2)
    read!("/scratch/fermi/jawla/2d_ising_training_data_L=$(args.lattice_size)_extremes.bin",df) #edit here for dataset
    len=size(df)[1]
    @info(@sprintf("%d is the number of samples",args.sample_size))
    split_boundary=ceil(Int64,0.9*len)
    @info(@sprintf("%f is the phase and %f is the temperature",df[2,args.n_features+2],df[2,args.n_features+1]))
	train=df[1:split_boundary,:]
    @info(@sprintf("%d is the size of training set",size(train)[1]))
	validate=df[split_boundary+1:end,:]
    x_train=permutedims(train[:,1:3])#args.n_features])
    x_train=Flux.normalise(x_train,dims=2)
    y_train=train[:,args.n_features+2]
    u_niq=length(unique(y_train))
    # One-hot-encode the labels
    y_train= onehotbatch(y_train, 0:u_niq-1) #make onehotbatch of lables, essentially columns of labels
    # Batching
    train_data = DataLoader(x_train, y_train, batchsize=args.batchsize,shuffle=true,partial=true)#makes an array of tuples(batches) where each batch is a matrix with columsn as samples, and rows as features

    x_validate=permutedims(validate[:,1:3])#args.n_features])
    x_validate=Flux.normalise(x_validate,dims=2)
    y_validate=validate[:,args.n_features+2]
    y_validate= onehotbatch(y_validate, 0:u_niq-1)
    @info(@sprintf("%d is the number of labels",u_niq))
    validate_data = DataLoader(x_validate, y_validate, batchsize=args.batchsize,shuffle=true,partial=true)#
    return train_data,validate_data,u_niq
end

function get_test_data(args)
    # Loading Dataset
	df=Array{Float32}(undef,args.test_sample_size,args.n_features+1)
    read!("/scratch/fermi/jawla/2d_ising_test_data_L=$(args.lattice_size).bin",df)
    x=permutedims(df[:,1:3])#args.n_features])
    x=Flux.normalise(x,dims=2)
    temps=permutedims(df[:,args.n_features+1])
    return x,temps
end

function weight_params(m::Chain, ps=Flux.params())
	map((l)->weight_params(l, ps), m.layers)
	ps
end

weight_params(m::Dense, ps=Flux.params()) = push!(ps, m.W)
weight_params(m::Conv, ps=Flux.params()) = push!(ps, m.weight)
weight_params(m::ConvTranspose, ps=Flux.params()) = push!(ps, m.weight)
weight_params(m, ps=Flux.params()) = ps
weight_params(m::Dropout, ps=Flux.params()) = ps

function build_model(args, nclasses)
    # return Chain(Dense(args.n_features, 256, relu),Dropout(0.25),BatchNorm(256),Dense(256,128,relu),Dropout(0.5),BatchNorm(128),Dense(128, nclasses))
	return Chain(Dense(3, 4, relu),BatchNorm(4),Dense(4, nclasses))
end

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Function to check if any element is NaN or not
anynan(x) = any(isnan.(x))

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += (sum(onecold(cpu(model(x))) .== onecold(cpu(y)))) / size(x,2)
    end
    acc/length(data_loader)
end


function train(; kws...)
    # Initializing Model parameters
    args = Args(; kws...)

    @info("Loading data set")
    train_set,validate_set,nclasses= getdata(args)

    # Construct model
    @info("Building model...")
    model = build_model(args,nclasses)
    #model = fmap(f64, model)
    # Load model and datasets onto GPU, if enabled
    train_set = args.device.(train_set)
    validate_set = args.device.(validate_set)
    model = args.device(model)

    # Make sure our model is nicely precompiled before starting our training loop
    #model(train_set[1][1])

    loss(x,y) = logitcrossentropy(model(x), y) + sum(sum(p.^2) for p in weight_params(model))
    # Train our model with the given training set using the ADAM optimizer

    opt = ADAM(args.η)


    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:args.epochs
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt )

        # Terminate on NaN
        if anynan(paramvec(model))
            @error "NaN params"
	    writedlm("activations.csv",Flux.activations(cpu(model),rand(args.n_features)))
	    paras=Array{Float32,1}[]
	    for i in params(model)
	    push!(paras,vec(i))
	    end
	    writedlm("parameters.csv",paras)
	    opt.eta /= 10.0
	    if epoch_idx == 5
               @warn(" -> Look for Division by zero")
               break
	    end
        end
        # Calculate accuracy:
        acc = accuracy(validate_set, model)

        @info(@sprintf("Epoch [%d]: validation set accuracy: %.4f", epoch_idx, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            # BSON.@save joinpath(args.savepath, "mlp_2d_ising_L=$(args.lattice_size).bson") params=cpu.(params(model)) epoch_idx acc
			a=cpu.(params(model))
			writedlm("./W1.csv",a[1])
			writedlm("./B1.csv",a[2])
			writedlm("./W2.csv",a[5])
			writedlm("./B2.csv",a[6])
	    break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc > best_acc
            @info(" -> New best accuracy! Saving model out to mlp_2d_ising.bson")
            BSON.@save joinpath(args.savepath, "mlp_2d_ising_L=$(args.lattice_size).bson") params=cpu.(params(model)) epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end
        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-9
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 10
            @warn(" -> We're calling this converged.")
            break
        end
    end
end

function test(; kws...)
    args = Args(; kws...)
    @info("Loading the test data")
    test_set,temp_list = get_test_data(args)
    # Re-constructing the model with random initial weights
    model = build_model(args,2)
    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "mlp_2d_ising_L=$(args.lattice_size).bson") params
    # Loading parameters onto the model
    Flux.loadparams!(model, params)
    testmode!(model)
    prob_dist=softmax(model(test_set))
    results=vcat(prob_dist,temp_list)
    writedlm("/home/jawla/Simulations/DNN_2d_ising_results_L=$(args.lattice_size)_extremes.csv",results)
end

cd(@__DIR__)
train()
test()