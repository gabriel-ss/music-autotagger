using Random

struct LazyDataLoader
	preprocess::Function
	datasource::Tuple
	batchsize::Int
	shuffle::Bool
	nobs::Int
	LazyDataLoader(preprocess, datasource; batchsize=1, shuffle=true, partial=false) = begin

		batchsize > 0 || throw(ArgumentError("Batchsize must be a positive number"))

		# Wrap the datasource in a tuple if isn't already
		datasource isa Tuple || (datasource = (datasource,))

		nobs = min(map(x->size(x)[end], datasource)...)
		if nobs < batchsize
			@warn "Number of data points less than batchsize, a single batch of $nobs points will be generated"
			batchsize = nobs
		end
		partial && (nobs -= nobs % batchsize)
		new(preprocess, datasource, batchsize, shuffle, nobs)

	end
end

LazyDataLoader(datasource; kwargs...) = LazyDataLoader(identity, datasource; kwargs...)
Base.length(loader::LazyDataLoader) = loader.nobs

function Base.iterate(loader::LazyDataLoader, (i, iterorder)=(1, nothing))

	i > loader.nobs && return nothing

	# During the first iteration define the access order of observations
	if (i === 1)
		iterorder = [1:loader.nobs;]
		loader.shuffle && shuffle!(iterorder)
	end

	# Get slices from data tuple following iterorder
	currentdata = map(loader.datasource) do data
		return view(
			data,
			(Base.Colon() for dim in 1:ndims(data)-1)...,
			iterorder[i:min(i + loader.batchsize - 1, loader.nobs)]
		)
	end |> loader.preprocess

	return (currentdata, ((i + loader.batchsize), iterorder))

end
