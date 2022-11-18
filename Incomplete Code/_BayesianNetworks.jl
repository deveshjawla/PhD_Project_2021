###
### Helper Functions
###

"""
Returns the output dimesions(except the last, batch size) of a convolution layer
"""
function conv_out_dims(input_dims::Tuple, cp::ConvParams)
    output_dims_conv = (((input_dims[1] - cp.filter_size[1] + 2 * cp.pad) / cp.stride_length) + 1)
    output_dims = ((output_dims_conv - cp.pool_window[1]) / cp.pool_stride) + 1
    return (Int(output_dims), Int(output_dims))
end

function num_params(cp::ConvParams)
    return prod([cp.filter_size..., cp.in_channels, cp.out_channels]) + cp.out_channels
end

function num_params(dp::DenseParams)
    return (dp.indim * dp.outdim) + dp.outdim
end

"""
Returns a tuple of the weight and bias arrays
"""
function layer_params(params_vec::AbstractVector, cp::ConvParams)
    bias = [pop!(params_vec) for _ in 1:cp.out_channels]
    weight = reshape(params_vec, cp.filter_size..., cp.in_channels, cp.out_channels)
    return weight, bias
end

function layer_params(params_vec::AbstractVector, dp::DenseParams)
    if dp.bias == true
        bias = [pop!(params_vec) for _ in 1:dp.outdim]
    else
        bias = [pop!(params_vec) for _ in 1:dp.outdim]
        # bias .*= 0.0
    end
    weight = reshape(params_vec, dp.outdim, dp.indim)
    return weight, bias
end

"""
Splits a vector x at indices n
"""
function split(x::AbstractVector, n)
    result = Vector{Vector{eltype(x)}}(undef, length(n))
    sum_elements = sum(n)
    if sum_elements == length(x)
        for i in 1:length(n)
            result[i] = splice!(x, 1:n[i])
        end
    end
    return result
end

"""
Make a list of tuples of layer parameters i.e [(W1,b1),...]
"""
function unpack_params(nn_params::AbstractVector, layers_spec::AbstractVector)
    num_params_list = num_params.(layers_spec)
    indices_list = split(nn_params, num_params_list)
    params_collection = [layer_params(i, j) for (i, j) in zip(indices_list, layers_spec)]
    return params_collection
end



###
### Conv Network specifications
###

# conv_layers = [ConvParams(), ConvParams(), ConvParams()]

# input_size = (128, 128)
# final_conv_out_dims = conv_out_dims(input_size, conv_layers[1]) |> x -> conv_out_dims(x, conv_layers[2]) |> x -> conv_out_dims(x, conv_layers[3])

# dense_layers = [DenseParams(indim = prod([final_conv_out_dims..., conv_layers[end].out_channels])), DenseParams(outdim = 10)]

# layers_spec = [conv_layers; dense_layers]

# function forward(x, nn_params::AbstractVector, layers_spec)
#     c1, c2, c3, d1, d2 = unpack_params(nn_params, layers_spec)
#     nn = Chain([layer(i..., j) for (i, j) in zip([c1, c2, c3], layers_spec[1:3])]..., Flux.flatten, [layer(i..., j) for (i, j) in zip([d1, d2], layers_spec[4:5])]..., softmax)
#     return nn(x)
# end

###
### Dense Network specifications
###

dense_layers = [DenseParams(indim = 150, outdim = 5, activation_fn = tanh), DenseParams(indim = 5, outdim = 3, activation_fn = tanh), DenseParams(indim = 3, outdim = 1, activation_fn = sigmoid)]

function forward(x, nn_params, layers_spec)
    nn = Chain([layer(i..., j) for (i, j) in zip(nn_params, layers_spec)]...)
    return nn(x)
end