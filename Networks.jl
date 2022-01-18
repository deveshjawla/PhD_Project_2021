using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON

function dense_layer(indim::Int, outdim::Int, bnmom::Float32, activation_fn)
    if typeof(bnmom) == Float32
        Chain(
            Dense(indim, outdim),
            BatchNorm(outdim, activation_fn, momentum = bnmom)
			)
    else
        Dense(indim, outdim, activation_fn)
    end
end

function conv_layer(input_dims, filter_size::Tuple{Int,Int}, i::Int, o::Int, activation_fn, stride_length, pool_window::Tuple{Int,Int}, bnmom::Float32)
    pad = size .÷ 2
    layer = Chain(
        Conv(filter_size, i => o, activation_fn, pad = pad),
        x -> meanpool(x, pool_window),
        BatchNorm(i, activation_fn, momentum = bnmom)
    )
	output_dims = (((input_dims[1] - filter_size[1] + 2*pad[1]) / stride_length) + 1)
    return layer, output_dims
end

