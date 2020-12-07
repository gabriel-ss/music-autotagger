using DSP, FFTW

"""
    melspectrogram(signal, sr, window=ones, windowsize=2048, overlap=512, nmels=26)

Compute the melspectrogram of the given signal.

**Parameters:**

`signal`: The signal on which the coefficients will be computed.

`sr`: Sample rate of the signal.

`window`: The window function to be applied on the signal by the STFT.

`windowsize`: The size of the STFT window.

`overlap`: The overlap of two adjacent windows of the STFT in samples.

`nmels`: The amount of mel bands to be extracted from the signal.
"""
function melspectrogram(
	signal, sr, window=ones, windowsize=2048, overlap=512, nmels=128
)

	# Transform the input to mel scale
	mel(frequency) = 1127*log(1 + frequency/700)

	# Transform the input back from mel scale
	imel(mel) = 700*(exp(mel/1127) - 1)

	# Return an array of mel spaced points
	melspace(start, length, stop) = imel.(range(mel(start), mel(stop), length=length))

	# Each column is a frame of stft
	sigstft = stft(signal, windowsize, overlap, window=window)

	# If the signal is real then the absolute value of it's spectrum is symmetric
	eltype(signal)<:Real && (windowsize = windowsize ÷ 2 + 1)

	# Evaluate the periodogram of the signal
	periodgrm = [abs2.(frame)/windowsize for frame in sigstft]

	# Create a set of linearly spaced points in the mel scale
	fmax = sr/2
	melpoints = ceil.(Integer, melspace(1, nmels + 2, fmax)*windowsize/fmax)

	# Create the filterbank...
	windows = map(1:nmels) do m
		# ...by creating a triangular window for each set of three consecutive points
		start, apex, stop = melpoints[m:m + 2]
		w = zeros(stop - start)

		for k in 1:(apex - start)
			w[k] = k/(apex - start)
		end

		for k in (apex + 1):stop
			w[k - start] = (stop - k)/(stop - apex)
		end

		return 2w/(stop - start)	# Normalized so that the triangle has area 1
	end

	windowstarts = melpoints[1:nmels]

	# Apply the filterbank to each frame of the periodogram...
	return mapslices(periodgrm, dims = 1) do frame
		# ...by multiplying the non zero part of each window in the filterbank
		return map(windows, windowstarts) do melwindow, winstart
			return melwindow'*frame[winstart:winstart + length(melwindow) - 1]
		end
	end

end
