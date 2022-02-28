import FileIO
import StatsBase
import DataFrames
import Random

function textonehot(df,col,delim=";")
	filtermut = [split(i,delim) for i in df[:,col]]
	filtermutect = unique(reduce(vcat,filtermut))
	filtermutectdata = DataFrame(zeros(Float64,nrow(df),length(filtermutect) ),string.(col,"_",filtermutect) )
	for i in 1:length(filtermut)
		for j in 1:length(filtermut[i]) # j is column to be set to 1
			id=findfirst(x->x == filtermut[i][j],filtermutect)
			filtermutectdata[i,id] = 1.0
		end
	end
	oldnames=names(filtermutectdata)
	nams = replace.(names(filtermutectdata),"."=>"_")
	diffs = oldnames .== nams
	for i in 1:length(nams)
		if ! diffs[i]
			DataFrames.rename!(filtermutectdata,oldnames[i]=>nams[i])
		end
	end
	return filtermutectdata
end

function sampler(xx2,yy2,train,testog,samplskey,way,ratio_under_to_over)
	if isapprox(ratio_under_to_over, 0)
		return xx2,yy2,train,testog,samplskey
	end
    xx=xx2[train,:]
    yy=yy2[train]
	smpsyy=samplskey[train]

    yy = [i==true for i in yy]
	truez = findall(x->x,yy)
	falsez = findall(x->!x,yy)
	u = length(truez)
	o = length(falsez)
	r =  u/o
    println("ratio: ",r)
	v = []
	if way == "over"
		x = round(Int,ratio_under_to_over * o - u)
        println("number of new: ",x)
		for i in StatsBase.sample(1:u,x,replace=true)
			push!(v, truez[i])
		end
        train = vcat(train,length(yy2)+1:length(yy2)+length(v))
		return vcat(xx2,xx[v,:]),vec(vcat(yy2,yy[v,:])),train,testog,vcat(samplskey,smpsyy[v])
	elseif way == "under"
		x = round(Int, u / ratio_under_to_over)
		println("number of new: ",x)
		for i in StatsBase.sample(1:o,x,replace=false)
			push!(v, falsez[i])
		end
		t1 = vcat(xx[v,:],xx[truez,:])
		train = collect(1:nrow(t1))
		# println(length(nrow(t1)+1))
		# println(nrow(xx2[testog,:]))
		# println(length(nrow(t1)+nrow(xx2[testog,:])))
		test = collect(nrow(t1)+1:nrow(t1)+nrow(xx2[testog,:]))
		t1 = vcat(t1,xx2[testog,:])
		t2 = vcat(yy[v],yy[truez],yy2[testog] )

		return t1,t2,train,test ,vcat(smpsyy[v],smpsyy[truez],samplskey[testog])
	else
		error("way not specified")
	end
	
end

function isonehot(val::Array)::Bool
	if length(val) > 3
		return false
	end
	for i in val
		if ! ismissing(i) && i > 1.1
			return false
		end
	end

	val = Set(val)
	if val == Set([missing,1.0,0.0])
		return true
	elseif val == Set([missing,1,0.0])
		return true
	elseif val == Set([missing,1.0,0])
		return true
	elseif val == Set([missing,1,0])
		return true
	elseif val == Set([missing,true,false])
		return true
	elseif val == Set([missing,1.0,false])
		return true
	elseif val == Set([missing,1,false])
		return true
	elseif val == Set([missing,true,0.0])
		return true
	elseif val == Set([missing,true,0])
		return true
	elseif val == Set([1.0,0.0])
		return true
	elseif val == Set([1,0.0])
		return true
	elseif val == Set([1.0,0])
		return true
	elseif val == Set([1,0])
		return true
	elseif val == Set([true,false])
		return true
	elseif val == Set([1.0,false])
		return true
	elseif val == Set([1,false])
		return true
	elseif val == Set([true,0.0])
		return true
	elseif val == Set([true,0])
		return true
	elseif val == Set([missing,0])
		return true
	elseif val == Set([missing,1])
		return true
	elseif val == Set([missing,0.0])
		return true
	elseif val == Set([missing,1.0])
		return true
	elseif val == Set([missing,true])
		return true
	elseif val == Set([missing,false])
		return true
	elseif val == Set([1])
		return true
	elseif val == Set([0])
		return true
	elseif val == Set([true])
		return true
	elseif val == Set([false])
		return true
	elseif val == Set([1.0])
		return true
	elseif val == Set([0.0])
		return true
	else
		return false
	end
end


struct standardizer
    col_names::Vector{String}
    means::Vector{Float64}
    stds::Vector{Float64}
end

struct normalizer
    col_names::Vector{String}
    maxs::Vector{Float64}
    mins::Vector{Float64}
end


function fit_standardizer(df)
	namevec = Vector{String}()
	meanvec = Vector{Float64}()
	stdvec = Vector{Float64}()
	for colname in names(df)
		if ! isonehot(unique(df[:,colname])) && typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Union{Float32, Float64}
			push!(namevec,colname)
			push!(meanvec,mean(skipmissing(df[:,colname])) )
			st = std(skipmissing(df[:,colname]))
			if st < eps(Float32)
				if typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Float32
					push!(stdvec, eps(Float32))
				else
					push!(stdvec, eps(Float64))
				end
			else
				push!(stdvec, st)
			end
		end
	end
	return standardizer(namevec,meanvec,stdvec)
end

function transform_standardizer!(df, standardzer)
	for col in 1:length(standardzer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,standardzer.col_names[col] ] = ( df[i,standardzer.col_names[col] ] - standardzer.means[col] ) / standardzer.stds[col]
		end
	catch
		continue
	end
	end
end

function transform_standardizer(df2, standardzer)
	df = copy(df2)
	for col in 1:length(standardzer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,standardzer.col_names[col] ] = ( df[i,standardzer.col_names[col] ] - standardzer.means[col] ) / standardzer.stds[col]
		end
	catch
		continue
	end
	end
	return df
end

function untransform_standardizer!(df, standardzer)
	for col in 1:length(standardzer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,standardzer.col_names[col] ] = ( df[i,standardzer.col_names[col] ] * standardzer.stds[col] ) + standardzer.means[col]
		end
	catch
		continue
	end
	end
end

function untransform_standardizer(df2, standardzer)
	df = copy(df2)
	for col in 1:length(standardzer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,standardzer.col_names[col] ] = ( df[i,standardzer.col_names[col] ] * standardzer.stds[col] ) + standardzer.means[col]
		end
	catch
		continue
	end
	end
	return df
end

function fit_normalizer(df)
	namevec = Vector{String}()
	maxvec = Vector{Float64}()
	minvec = Vector{Float64}()
	for colname in names(df)
		if ! isonehot(unique(df[:,colname])) && typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Union{Float32, Float64}
			push!(namevec,colname)
			mx = maximum(skipmissing(df[:,colname]))
			push!(maxvec,mx )
			st = minimum(skipmissing(df[:,colname]))
			if isapprox(mx,st)
				if typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Float32
					push!(minvec, st-eps(Float32))
				else
					push!(minvec, st-eps(Float64))
				end
			else
				push!(minvec, st)
			end
		end
	end
	return normalizer(namevec,maxvec,minvec)

end

function transform_normalizer!(df, normalizer)
	for col in 1:length(normalizer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,normalizer.col_names[col] ] = ( df[i,normalizer.col_names[col] ] - normalizer.mins[col] ) / ( normalizer.maxs[col] - normalizer.mins[col]  )
		end
	catch
		continue
	end
	end
end

function transform_normalizer(df2, normalizer)
	df = copy(df2)
	for col in 1:length(normalizer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,normalizer.col_names[col] ] = ( df[i,normalizer.col_names[col] ] - normalizer.mins[col] ) / ( normalizer.maxs[col] - normalizer.mins[col]  )
		end
	catch
		continue
	end
	end
	return df
end

function untransform_normalizer!(df, normalizer)
	for col in 1:length(normalizer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,normalizer.col_names[col] ] = df[i,normalizer.col_names[col] ] * ( normalizer.maxs[col] - normalizer.mins[col]  ) + normalizer.mins[col]  
		end
	catch
		continue
	end
	end
end

function untransform_normalizer(df2, normalizer)
	df = copy(df2)
	for col in 1:length(normalizer.col_names)
		try
		for i in 1:DataFrames.nrow(df)
			df[i,normalizer.col_names[col] ] = df[i,normalizer.col_names[col] ] * ( normalizer.maxs[col] - normalizer.mins[col]  ) + normalizer.mins[col]  
		end
	catch
		continue
	end
	end
	return df
end



function savechunk(df::DataFrame,sample_key::Vector{String},name::String;overwrite=false::Bool,chunk_size=5000::Int)::Nothing
	try
		mkdir(name)
	catch
		if overwrite
			println("File already exists: overwriting")
			rm(name, recursive=true)
			mkdir(name)
		else
			println("File already exists: not overwriting")
			return nothing
		end
	end
	ids = collect(Iterators.partition(collect(1:nrow(df)), chunk_size))
	for j in 1:length(ids)
		FileIO.save("$name/$j.jld2", Dict("sample_key"=>sample_key[ids[j]],"x" => x[ids[j],:],
			"y" => y[ids[j]], 
			#         "train" => train, "test" => test
			) ; compress = true
		)
	end
	return nothing
end	

function loadchunk(name::String)
	try
		readdir(name)
	catch
		println("File does not exist")
		return nothing
	end
	ids = readdir(name)
	sample_key, xold, yold = FileIO.load("$name/1.jld2","sample_key","x" ,"y" );
	for j in 1:length(ids)
		if occursin(".jld2",ids[j]) 
			n = parse(Int,split(ids[j],".")[1])
			if n > 1
				sample_keyt, xoldt, yoldt = FileIO.load("$name/$n.jld2","sample_key","x" ,"y" );
				sample_key = vcat(sample_key,sample_keyt)
				xold = vcat(xold,xoldt)
				yold = vcat(yold,yoldt)
			end
		end
	end
	return sample_key, xold, yold
end


function logger(x::Missing,base::Real)::Missing
    return x
end

function logger(x::Float32,base::Real)::Float32
	if base <= 0.0
		error("base is not positive")
	end
    if x < 1.0f0 # 0.0000000001f0
        return 0.5f0*x-0.5f0 #log(base,0.0000000001f0)
    else
        return log(base,x)
    end
end


function unlogger(x::Missing,base=ℯ::Float64)::Missing
    return x
end

function unlogger(x::Float32,base::Real)::Float32
	if base <= 0.0
		error("base is not positive")
	end
	if x < 1.0f0
		return 2.0f0*(x+0.5f0)
	end
    return base^x
end


function transform_log!(df, base)
	for colname in names(df)
		if ! isonehot(unique(df[:,colname])) && typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Union{Float32, Float64}
			df[:,colname ] = logger.(df[:,colname ],base)
		end
	end
end

function transform_log(df2, base)
	df = copy(df2)
	for colname in names(df)
		if ! isonehot(unique(df[:,colname])) && typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Union{Float32, Float64}
			df[:,colname ] = logger.(df[:,colname ],base)
		end
	end
	return df
end

function untransform_log!(df, base)
	for colname in names(df)
		if ! isonehot(unique(df[:,colname])) && typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Union{Float32, Float64}
			df[:,colname ] = unlogger.(df[:,colname ],base)
		end
	end
end

function untransform_log(df2, base)
	df = copy(df2)
	for colname in names(df)
		if ! isonehot(unique(df[:,colname])) && typeof(df[findfirst(x->!ismissing(x),df[:,colname]),colname]) <: Union{Float32, Float64}
			df[:,colname ] = unlogger.(df[:,colname ],base)
		end
	end
	return df
end


function scrambler(df2,seed=0)
	df = copy(df2)
	Random.seed!(seed)
	tt = split.(df.sample_key," ")
	samples = [i[1] for i in tt]
	# chrom = [string(i[2][4:end]) for i in tt]
	# pos = [string(i[3]) for i in tt]
	key = [i[2]*" "*i[3] for i in tt]
	mut = [string(i[4]) for i in tt]

	samples_uniq = unique(samples)
	keys_uniq = unique(key)
	samples_dict = Dict()
	key_dict = Dict()
	for samp in samples_uniq
		samples_dict[samp] = Random.randstring('A':'Z',1)*Random.randstring(['a','e','i','o','u'],1)*Random.randstring('a':'z',1)*Random.randstring('1':'9',2)
	end
	for samp in keys_uniq
		v = "chr$(rand(25:50)) $(rand(100000:9999999))"
		key_dict[samp] = v
	end
	df.sample_key = copy(df.sample_key)
	df.key = copy(df.sample_key)
	df.SAMPLE = copy(df.sample_key)
	for i in 1:nrow(df)
		df.sample_key[i] = string( samples_dict[samples[i]], " ", key_dict[key[i]]," ",mut[i] )
		df.key[i] = string( key_dict[key[i]] )
		df.SAMPLE[i] = string( samples_dict[samples[i]] )
	end
	return (df,samples_dict,key_dict)
end

function unscrambler(df2,samples_dict,key_dict)
	df = copy(df2)
	tt = split.(df.sample_key," ")
	samples = [i[1] for i in tt]
	key = [i[2]*" "*i[3] for i in tt]
	mut = [string(i[4]) for i in tt]
	samples_dict_inv = Dict()
	key_dict_inv = Dict()
	for (k,v) in samples_dict
		samples_dict_inv[v] = k
	end
	for (k,v) in key_dict
		key_dict_inv[v] = k
	end
	for i in 1:nrow(df)
		df.sample_key[i] = string( samples_dict_inv[samples[i]], " ", key_dict_inv[key[i]]," ",mut[i] )
		df.key[i] = string( key_dict_inv[key[i]] )
		df.SAMPLE[i] = string( samples_dict_inv[samples[i]] )
	end
	return (df)
end
