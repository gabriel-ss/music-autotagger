#!/bin/bash
#=
project=$(dirname $0)/..
sysimage=$project/sysimage.so

if [ ! -e "$sysimage" ]; then
	# echo not found
	julia --project="$project" "$project/src/precompile.jl"
fi

exec julia --project=$project --sysimage="$sysimage" --startup-file=no \
    --color=yes -e 'include(popfirst!(ARGS))' "${BASH_SOURCE[0]}" "$@"
=#

using Logging, Dates
include("./preprocess/preprocess.jl")
include("./training/training.jl")
include("./LazyDataLoader.jl")
include("./cliparser.jl")

function main(options)

	log_io = open("train-$(now()).log", "w+")
	global_logger(SimpleLogger(log_io))

	feature_shape = Dict(
		"chroma" => (12, 582),
		"melspectrogram" => (128, 582),
		"mfcc" => (26, 582),
	)[options["selectedfeature"]]


	if options["shufflesamples"]
		shuffle_rows(options["tagsfile"])
	end


	if options["preprocess"]
		@info "Preprocessing audio..."
		preprocess_audio(options["selectedfeature"], options["tagsfile"],
			options["pathcolumn"], options["samplespath"],
			options["audiofeaturesdir"])
	end


	if options["trainnetwork"]
		(train_set, validation_set, test_set), tagnames =
			preprocess_tags(options["tagsfile"], options["pathcolumn"],
				options["excludedcolumns"], partitions=(13, 1, 2))

		for set in (train_set, validation_set, test_set)
			# Append features path and remove file extension
			for (i,) in enumerate(set[1])
				set[1][i] = joinpath(options["audiofeaturesdir"], set[1][i])[1:end-4]
			end
		end

		# Create data loader to load data from disk to gpu on demand
		train_set = LazyDataLoader(create_feature_loader(feature_shape), train_set, batchsize=options["batchsize"])

		@info "Loading validation set..."
		validation_set = (load_features(validation_set[1], feature_shape), copy(validation_set[2]))

		train_cnn(feature_shape, train_set, validation_set,
			options["modeloutputdir"], options["epochs"], log_io, options["model"])
	end

	close(log_io)

end

function load_features(filepaths, feature_shape)
	features = Array{Float32}(undef, feature_shape..., length(filepaths))
	for (i, filepath) in enumerate(filepaths)
		features[:, :, i] = reshape(reinterpret(Float32, read(filepath)), feature_shape...)
	end

	return features
end

function create_feature_loader(feature_shape)
	return ((filepaths, tags),) -> gpu((load_features(filepaths, feature_shape), copy(tags)))
end


main(options)
