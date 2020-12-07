include("./chroma.jl")
include("./melspectrogram.jl")
include("./mfcc.jl")

using CSV, DataFrames, MP3, DSP, Statistics, Random, ProgressMeter


function shuffle_rows(tagsfile)

	annotations = CSV.File(tagsfile)
	new_order = shuffle(1:size(annotations)[1])
	annotations = annotations[new_order]
	CSV.write(tagsfile, annotations)

end


function preprocess_tags(tagsfile, pathcolumn, excludedcolumns; partitions=nothing)

	annotations = DataFrame(CSV.File(tagsfile))

	# Sort the tags by frequency...
	tag_frequencies = [
		column => sum(annotations[:, column])
		for column in names(annotations) if column ∉ [pathcolumn, excludedcolumns...]
	]
	sort!(tag_frequencies, lt = (pair1, pair2) -> pair1.second > pair2.second)

	#  ...and get the 50 most frequent tags
	select!(annotations, All(pathcolumn, [pair.first for pair in tag_frequencies[1:50]]))

	datasetsize, = size(annotations)
	filepaths = Array(select(annotations, pathcolumn))[:]
	tags = permutedims(Array(select!(annotations, Not(pathcolumn))))
	tagnames = names(annotations)

	# If a partition tuple is specified, divide the dataset in the given proportions
	if partitions isa Tuple
		partitionsizes = [datasetsize * partition ÷ sum(partitions)
			for partition in partitions]

		partitionends = cumsum(partitionsizes)
		partitionstarts = [0; partitionends[1:end-1]] .+ 1

		return (map(partitionstarts, partitionends) do pstart, pend
			return (filepaths[pstart:pend], tags[:, pstart:pend])
		end, tagnames)
	end

	return ((filepaths, tags), tagnames)

end


function preprocess_audio(feature_type, tagsfile, pathcolumn, samplespath, outputpath)

	annotations = [row[Symbol(pathcolumn)] for row in CSV.File(tagsfile, select=[pathcolumn])]
	progress = Progress(length(annotations))

	for track in annotations
		# Pre-processes the audio...
		audio = load(joinpath(samplespath, track))
		feature = feature_list[feature_type](audio.data[:], audio.samplerate)
		normalized_feature = normalize_feature(feature)

		# ...and write the result to file
		(trackdir, ) = splitdir(track)
		mkpath(joinpath(outputpath, trackdir))
		write(joinpath(outputpath, track[1:end-4]), normalized_feature)
		next!(progress)
	end

end

function normalize_feature(feature)
	featuremean = mean(feature[:])
	return Float32.(feature .- featuremean ./ std(feature, mean=featuremean) .+ eps(Float32))
end

feature_list = Dict(
	"chroma" => (signal, sr) ->
		chroma(signal, sr, hanning, Int(sr ÷ 20), 0, 0.1, 0.1),
	"melspectrogram" => (signal, sr) ->
		melspectrogram(signal, sr, hanning, Int(sr ÷ 20), 0, 128),
	"mfcc" => (signal, sr) ->
		mfcc(signal, sr, hanning, Int(sr ÷ 20), 0, 26),
)
