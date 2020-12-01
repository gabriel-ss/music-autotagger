include("./melspectrogram.jl")
using FFTW

"""
    mfcc(signal, sr, window=ones, windowsize=2048, overlap=512, numcep=26)

Compute the Mel-Frequency Cepstral Coefficients of the given signal and return
an Array where each column contains the coefficients of a given frame obtained
from the stft.

**Parameters:**

`signal`: The signal on which the coefficients will be computed.

`sr`: Sample rate of the signal.

`window`: The window function to be applied on the signal by the STFT.

`windowsize`: The size of the STFT window.

`overlap`: The overlap of two adjacent windows of the STFT in samples.

`numcep`: The amount of cepstral coefficients to be extracted from the signal.
"""
function mfcc(
	signal, sr, window=ones, windowsize=2048, overlap=512, numcep=26
)

	# Evaluate the melspectrogram
	melspec = melspectrogram(signal, sr, window, windowsize, overlap, numcep)

	#Take the Type-III DCT of the log for each coeficient
	return mapslices(frame->idct(log.(frame)), melspec, dims=1)

end
