using Flux, BSON, Printf, Dates
using Flux: crossentropy, throttle

include("./roc.jl")
include("./dashboard.jl")
include("./model_definitions.jl")


function train_cnn(feature_shape, train_set, validation_set, modeloutputdir, n_of_epochs, log_io, model_path=nothing)

	if model_path === nothing
		model = create_model(feature_shape...)
	else
		BSON.@load model_path model
	end
	model = gpu(model)

	optmizer = ADAM(1E-3)
	paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

	function accuracy(x, y)
		ŷ = cpu(model.(reshape(s, size(s)..., 1) for s in eachslice(x |> gpu, dims=3)))
		ŷ = reduce(hcat, ŷ)
		rows = size(y)[1]
		acc = Vector{Float64}(undef, rows)
		average_acc = 0
		for i in 1:rows
			acc[i] = auc(roc(ŷ[i ,:], y[i ,:])[1:2]...)
		end
		return acc
	end
	loss(x, y) = Flux.Losses.logitbinarycrossentropy(model(x), y)


	mkpath(modeloutputdir)
	acc_dump = open("acc_dump$(now())", "w+")
	dashboard_height, dashboard_width = (20, 80)
	mean_acc = 0.0
	last_improvement = 0
	accuracy_curve = Vector{Float64}()
	target_acc = 0.9
	best_mean_acc = 0.0
	last_improvement = 0
	print("\u001b[0J\r", '\n'^(dashboard_height + 4))


	try
		@info("Beginning training loop...")
		start_time = now()
		for current_epoch in 1:n_of_epochs

			# Train for a single epoch
			Flux.train!(loss, params(model), train_set, optmizer)

			if any(isnan.(paramvec(model)))
				@error "NaN params"
				break
			end

			# Calculate accuracy:
			acc = accuracy(validation_set...)
			write(acc_dump, acc)
			mean_acc = sum(acc) / length(acc)
			@info(@sprintf("[%d]: Test accuracy: %.4f", current_epoch, mean_acc))

			if mean_acc >= best_mean_acc
				 @info(" -> New best accuracy")
				 best_mean_acc = mean_acc
				 last_improvement = current_epoch
			end

			print("\r\u001b[$(dashboard_height + 2)A")
			show_dashboard(current_epoch, n_of_epochs, start_time,
				push!(accuracy_curve, mean_acc), target_acc, best_mean_acc,
				height=dashboard_height, width=dashboard_width,)

			BSON.@save joinpath(modeloutputdir, "model-$mean_acc-$(now()).bson") model=cpu(model) current_epoch mean_acc

			if mean_acc >= target_acc
				@info("Target accuracy reached")
				break
			end

			# If no improvement has been made in the last 10% of the total training time, drop the learning rate:
			if current_epoch - last_improvement >= n_of_epochs ÷ 10 && optmizer.eta > 1E-9
				optmizer.eta /= 10.0
				@warn("Haven't improved in a while, dropping learning rate to $(optmizer.eta)")

				# After dropping learning rate, give it a few epochs to improve
				last_improvement = current_epoch
			end

			if current_epoch - last_improvement >= n_of_epochs ÷ 10
				@warn("We're calling this converged.")
				break
			end
		end
	catch err
		if err isa InterruptException
			@warn "Interrupted by user"
		else
			throw(err)
		end

	finally
		close(acc_dump)
	end


end
