#### 
#### Network - A customizable chain of conv layer and dense layers. Does not require manually
#### calculating the output size of images. Takes as input batches of images with dims WHCN.
#### 

using Flux, Statistics
using Printf, BSON

"""
# Function to make a custom dense layer
"""
function dense_layer(indim::Int, outdim::Int,
	bnmom::Float32, activation_fn)

    if typeof(bnmom) == Float32
        Chain(
            Dense(indim, outdim),
            BatchNorm(outdim, activation_fn, momentum = bnmom)
			)
    else
        Dense(indim, outdim, activation_fn)
    end
end

"""
# Fucntion to make a custom Covolution layer.
## Returns a tuple, (layer, output_image_dims)
"""
function conv_layer(input_dims, filter_size::Tuple{Int,Int}, 
	i::Int, o::Int, activation_fn, pool_window::Tuple{Int,Int}; 
	stride_length=1, pool_stride=1, bnmom=0.5)

    pad = filter_size .÷ 2
	if typeof(bnmom) == Float32
		layer = Chain(
			Conv(filter_size, i => o, activation_fn, pad = pad),
			x -> meanpool(x, pool_window),
			BatchNorm(i, activation_fn, momentum = bnmom)
		)
	else
		layer = Chain(
			Conv(filter_size, i => o, activation_fn, pad = pad),
			x -> meanpool(x, pool_window, stride= pool_stride)
			)
	end
    output_dims_conv = (((input_dims[1] - filter_size[1] + 2 * pad[1]) / stride_length) + 1)
	output_dims = ((output_dims_conv - pool_window[1])/pool_stride) + 1
    return layer, Int(output_dims)
end

conv_1, out_1 = conv_layer(128, (3,3), 1, 1, relu, (2,2), bnmom=nothing)
conv_2, out_2 = conv_layer(out_1, (3,3), 1, 1, relu, (2,2), bnmom=nothing)
conv_3, out_3 = conv_layer(out_2, (3,3), 1, 1, relu, (2,2), bnmom=nothing)
dense_1 = dense_layer(out_3^2, 128, 0.1f0, relu)
dense_2 = dense_layer(128, 10, 0.1f0, relu)

# An example Network. You may customize each layer above or remove/add layers as needed.
network = Chain(conv_1, conv_2, conv_3, flatten, dense_1, dense_2, softmax)
# network(rand(128,128,1,1))