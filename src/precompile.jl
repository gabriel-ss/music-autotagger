using PackageCompiler

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
	project=joinpath(@__DIR__, ".."),
	sysimage_path=joinpath(@__DIR__, "../sysimage.so"))
