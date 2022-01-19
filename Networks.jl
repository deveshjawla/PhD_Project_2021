#### 
#### Network - A customizable chain of conv layer and dense layers. Does not require manually
#### calculating the output size of images. Takes as input batches of images with dims WHCN.
#### 

using Flux, Statistics
using Printf, BSON

"""
# Function to make a custom dense layer
"""
function dense_layer(indim::Int32, outdim::Int32,
	bnmom::Float32, activation_fn)

   
    return Chain(
            Dense(indim, outdim),
            BatchNorm(outdim, activation_fn, momentum = bnmom)
			)
    
end

function dense_layer(indim::Int32, outdim::Int32,
	bnmom::Nothing, activation_fn)

    
    return Dense(indim, outdim, activation_fn)
   
end

"""
# Fucntion to make a custom Covolution layer.
## Returns a tuple, (layer, output_image_dims)
"""
function conv_layer(input_dims, filter_size::Tuple{Int32,Int32}, 
	i::Int32, o::Int32, activation_fn, pool_window::Tuple{Int32,Int32}; 
	stride_length::Int32=Int32(1), pool_stride::Int32=Int32(1), bnmom::Float32=0.5f0)

    pad = filter_size .÷ Int32(2)
	
	layer = Chain(
		Conv(filter_size, i => o, activation_fn, pad = pad),
		x -> meanpool(x, pool_window),
		BatchNorm(i, activation_fn, momentum = bnmom)
	)
	
    output_dims_conv = (((input_dims[1] - filter_size[1] + Int32(2) * pad[1]) / stride_length) + Int32(1))
	output_dims = ((output_dims_conv - pool_window[1])/pool_stride) + Int32(1)
    return layer, Int(output_dims)
end

function conv_layer(input_dims, filter_size::Tuple{Int32,Int32}, 
	i::Int32, o::Int32, activation_fn, pool_window::Tuple{Int32,Int32}; 
	stride_length::Int32=Int32(1), pool_stride::Int32=Int32(1), bnmom::Nothing)

    pad = filter_size .÷ Int32(2)
	
	layer = Chain(
		Conv(filter_size, i => o, activation_fn, pad = pad),
		x -> meanpool(x, pool_window, stride= pool_stride)
		)
	
    output_dims_conv = (((input_dims[1] - filter_size[1] + Int32(2) * pad[1]) / stride_length) + Int32(1))
	output_dims = ((output_dims_conv - pool_window[1])/pool_stride) + Int32(1)
    return layer, Int(output_dims)
end

conv_1, out_1 = conv_layer(Int32(128), (Int32(3),Int32(3)), Int32(1), Int32(1), relu, (Int32(2),Int32(2)), Int32(1),bnmom=nothing)
conv_2, out_2 = conv_layer(Int32(out_1), (Int32(3),Int32(3)), Int32(1), Int32(1), relu, (Int32(2),Int32(2)), bnmom=nothing)
conv_3, out_3 = conv_layer(Int32(out_2), (Int32(3),Int32(3)), Int32(1), Int32(1), relu, (Int32(2),Int32(2)), bnmom=nothing)
dense_1 = dense_layer(Int32(out_3^2), Int32(128), 0.1f0, relu)
dense_2 = dense_layer(Int32(128), Int32(10), 0.1f0, relu)

# An example Network. You may customize each layer above or remove/add layers as needed.
network = Chain(conv_1, conv_2, conv_3, flatten, dense_1, dense_2, softmax)
# network(rand(128,128,1,1))