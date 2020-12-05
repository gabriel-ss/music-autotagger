using PackageCompiler

println("Compiling sysimage...")
create_sysimage([
		:ArgParse,
		:BSON,
		:CSV,
		:DSP,
		:DataFrames,
		:FFTW,
		:Flux,
		:MP3,
		:ProgressMeter,
		:UnicodePlots,
	];
	project="../",
	sysimage_path="../sysimage.so")
