 using Statistics

"""
# Gramian Angular Field
## Converts a timeseries into an image

Parameters
----------
image_size : int
    Shape of the output images.
	 
sample_range : tuple (min, max) (default = (-1, 1))
    Desired range of transformed data. If None, no scaling is performed
    and all the values of the input data must be between -1 and 1.
    If tuple, each sample is scaled between min and max; min must be
    greater than or equal to -1 and max must be lower than or equal to 1.

method : 'summation' or 'difference' (default = 'summation')
    Type of Gramian Angular Field. 's' can be used for 'summation'
    and 'd' for 'difference'.

flatten : bool (default = False)
    If True, images are flattened to be one-dimensional.

References
----------
.. [1] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
       Inspection and Classification Using Tiled Convolutional Neural
       Networks." AAAI Workshop (2015)
"""

function gramian_angular_field(input::Int32,
    image_size::Int32 = Int32(1),
    sample_range::Tuple{Int32,Int32} = (-Int32(1),Int32(1)),
    method::Symbol = :Summation,
    flatten::Bool = false)

	_, X_paa = paa(input, image_size)
	X_cos, X_sin = X_cos_sin_tuple(X_paa, sample_range)

	if method === :Summation
		output = X_cos*X_cos' - X_sin*X_sin'
	elseif method === :Difference
		output = X_sin*X_cos' - X_cos*X_sin'
	else
		error("Not a valid method")
	end
				
    return flatten ? reshape(output,:) : output
end

# function split(x::Vector, n)
#      result = Vector{Vector{eltype(x)}}()
#      for i in 1:length(n)
#          if length(x) < n[i]
#              push!(result, [])
#          elseif i==1
#              push!(result, x[1:n[i]])
#          else
#              push!(result, x[n[i-1]+1:n[i]])
#          end
#      end
#      result
# end
function split(x::Vector, n)
	result = Vector{Vector{eltype(x)}}(undef,length(n))
	for i in 1:length(n)
		if length(x) < n[i]
			result[i] = []
		elseif i==Int32(1)
			result[i] = x[1:n[i]]
		else
			result[i] = x[n[i-1]+1:n[i]]
		end
	end
	return result
end


using StatsBase
function paa(input_vector, output_size)
	if length(input_vector)%output_size == Int32(0)
		quotient_ = Int(length(input_vector)/output_size)
		indices_list = [((i-1)*quotient_)+1:quotient_*i for i in 1:output_size]
		output_vector = [mean(indices) for indices in indices_list]
		output_vector_magnitude = [mean([input_vector[i] for i in indices]) for indices in indices_list]
	else
		value_space = 1:(length(input_vector)*output_size)
		output_indices = cld.(value_space, length(input_vector))
		input_indices = cld.(value_space, output_size)
		count_map = countmap(output_indices)
		uniques, nUniques = keys(count_map), values(count_map)
		output_vector_magnitude = [mean([input_vector[i] for i in indices]) for indices in split(input_indices, cumsum(nUniques))]
		output_vector = [mean(indices) for indices in split(input_indices, cumsum(nUniques))]
	end
    return output_vector, output_vector_magnitude
end

function X_cos_sin_tuple(X_paa, sample_range::Tuple{Int32,Int32})
	
	X_cos = minmaxscaling(X_paa, sample_range)
	
	X_sin = sqrt.(clamp.(Int32(1) .- X_cos.^2, Int32(0), Int32(1)))
	return X_cos, X_sin
end

function X_cos_sin_tuple(X_paa, sample_range::Nothing)

	if (X_min < -Int32(1)) || (X_max > Int32(1))
		error("If 'sample_range' is None, all the values of X must be between -1 and 1.")
	end
	X_cos = X_paa

	X_sin = sqrt.(clamp.(Int32(1) .- X_cos.^2, Int32(0), Int32(1)))
	return X_cos, X_sin
end

function minmaxscaling(input::Vector, range::Tuple)
	min, max = extrema(input)
	clip_min = range[1]
	clip_max = range[2]
	return ((input .- min)/(max-min)) * (clip_max - clip_min) .+ clip_min
end