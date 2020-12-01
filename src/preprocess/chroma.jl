"""
    chroma(signal, sr, window=ones, windowsize=2048, overlap=512, γ=0.1, ϵ=0.1)

Compute the chroma features of a signal.

The return will be a matrix of size 12xN where N is the number of frames of the
STFT and each column is a vector of normalized chroma features.


**Parameters:**

`signal`: The signal on which the coefficients will be computed.

`sr`: The sampling rate of the signal.

`window`: The window function to be applied on the signal by the STFT.

`windowsize`: The size of the STFT window.

`overlap`: The overlap of two adjacent windows of the STFT in samples.

`γ`: The compression factor to be used in the chroma features extraction.

`ϵ`: The smallest norm that a chroma vector needs to have to not be replaced by
a stub one.
"""
function chroma(signal, sr, window=ones, windowsize=2048, overlap=512, γ=0.1, ϵ=0.1)

	sigstft = stft(signal, windowsize, overlap, window=window)

	# If the signal is real then the absolute value of it's spectrum is symmetric
	eltype(signal)<:Real && (windowsize = windowsize ÷ 2 + 1)

	# The number of chroma vectors is defined by the frames in the STFT
	nofframes = size(sigstft)[2]
	pitchspectrum = Matrix{Real}(undef, 128, nofframes)

	# Relation of a pitch with a frequency defined by the MIDI standard
	pitch2freq(p) = round(Cint, (windowsize/sr)*2^((p - 69)/12)*440)

	# The bound frequencies of each one of the pitch classes
	bounds = [1 + pitch2freq(p - 0.5) for p in 0:128]

	for n in 1:nofframes
		# The pitch spectrum is a log spaced spectrum where a component is
		# proportional to the sum of the power of all stft components inside the
		# bounds of a pitch class
		pitchspectrum[:, n] =
			[sum(abs2.(sigstft[bounds[i]:bounds[i + 1],n])) for i in 1:128]
	end

	chroma = Matrix{Real}(undef, 12, nofframes)

	for n in 1:nofframes
		# A chroma vector is given by the sum of the pitch classes of the same
		# note in all octaves
		chroma[:, n] =
			[sum(pitchspectrum[(class % 12 + 1):12:end, n]) for class in 0:11]
	end

	# Compresses the chroma vectors
	chroma = log10.(1 .+ γ .* chroma)

	stubFeature = ones(12)./√12

	# Evaluates the norm of each chroma vector...
	for i in 1:nofframes
		# ...and if it's smaller than the ϵ parameter normalizes the vector
		if ((norm = √(sum(chroma[:, i].^2))) > ϵ)
			chroma[:, i] ./= norm
		# Otherwise replaces the vector with a stub one
		else
			chroma[:, i] = stubFeature
		end
	end

	return chroma

end
