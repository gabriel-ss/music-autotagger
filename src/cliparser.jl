include("./options.jl")
using ArgParse

function parse_commandline()

	settings = ArgParseSettings()

	@add_arg_table! settings begin
		"file"
			help = "audio file to be classified or dataset config file"
			required = false

		"--shufflesamples", "-s"
			help = "shuffles the samples in the dataset and save the result in " *
						"the csv file (do not use with previously started training " *
						"or the test set will be tainted with training and " *
						"validation samples)"
			action = :store_true
		"--trainnetwork", "-t"
			help = "train a network with the dataset described by the csv file"
			action = :store_true
		"--preprocess", "-p"
			help = "extract features from mp3 audio files to be used during the " *
					"network training, overwriting existing extracted feature " *
					"files (note: can be used with the -t flag to start training " *
					"after pre-processing)"
			action = :store_true
		"--selectedfeature", "-f"
			help = "define the features that will be extracted from mp3 audio " *
					"files and/or that will be used during the network training. " *
					"Can be set to 'chroma', 'melspectrogram' or 'mfcc'."
			arg_type = String
			default = "melspectrogram"
			range_tester = arg -> arg ∈ ["chroma", "melspectrogram", "mfcc"]
		"--audiofeaturesdir", "-d"
			help = "the directory to save extracted features after preprocessing " *
					"and to read features before training, will be appended to the " *
					"paths found in the CSV file to generate a filepath for each " *
					"item in the dataset"
			arg_type = String
			default = "./preprocessed_audio"
		"--model", "-m"
			help = "the trained model to be used to classify a music, or to be " *
					"trained"
			arg_type = String
		"--modeloutputdir", "-o"
			help = "the directory to save model parameters learned during the " *
					"training process, flags that should be ignored"
			arg_type = String
			default = "./models"
		"--epochs", "-e"
			help = "the number of training epochs"
			arg_type = Int
			default = 20
		"--batchsize", "-b"
			help = "the number of samples per batch"
			arg_type = Int
			default = 5

	end

	return parse_args(settings)

end

merge!(options, parse_commandline())
