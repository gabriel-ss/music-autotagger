using Flux

function create_model(feature_height, feature_width)

	return Chain(
		input -> reshape(input, size(input)[1:2]..., 1, size(input)[3]),
		# Conv feature_height×4×256 (ReLU, Output: 1×589×256)
		Conv((feature_height, 4), 1 => 256, relu, pad=(0, 5)),
		# Max Pooling 2 (Output: 294×256)
		MaxPool((1, 2)),
		# Conv 4×256 (ReLU, Output: 291×256)
		Conv((1, 4), 256 => 256, relu),
		# Batch Normalization
		BatchNorm(256),
		# Max Pooling 2 (Output: 145×256)
		MaxPool((1, 2)),
		# Conv 4×384 (ReLU, Output: 142×384)
		Conv((1, 4), 256 => 384, relu),
		# Max Pooling 2 (Output 71×384)
		MaxPool((1, 2)),
		# Batch Normalization
		BatchNorm(384),
		# Conv 71×50 (Sigmoid, Output: 50×1)
		Conv((1, 71), 384 => 50, sigmoid),
		input -> reshape(input, getindex(size(input), [3, 4])...),
	)

end
