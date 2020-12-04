using UnicodePlots, Dates

shiftprintln(shift) = x -> print("\u001b[$(shift)G\u001b[0K$(x)\u001b[1B")


"""
    show_dashboard(current_epoch, n_of_epochs, start_time, accuracy_curve,
	 	target_accuracy, best_accuracy=max(accuracy_curve...);
	 	height=min(20, displaysize(stdout)[1]),
		width=min(80, displaysize(stdout)[2]))

Show information about the current state of the training process.
"""
function show_dashboard(current_epoch, n_of_epochs, start_time,
	accuracy_curve, target_accuracy, best_accuracy=max(accuracy_curve...);
	height=min(20, displaysize(stdout)[1]), width=min(80, displaysize(stdout)[2]))

	plot_height = height - 3
	plot_width = width - 37
	elapsed_time = now() - start_time
	estimated_time = (elapsed_time * (n_of_epochs - current_epoch)) ÷ current_epoch

	p = lineplot(accuracy_curve, xlabel="Epoch", ylabel="Accuracy", color=:green,
		xlim=(1, n_of_epochs), ylim=(0, target_accuracy), width=plot_width, height=plot_height)
	lines!(p, 0, best_accuracy, n_of_epochs, best_accuracy, color=:blue)
	lines!(p, 0, target_accuracy, n_of_epochs, target_accuracy, color=:magenta)

	show(p)
	print("\r\u001b[$(plot_height + 2)A")
	shiftedprintln = shiftprintln(plot_width + 21)

	shiftedprintln("Elapsed Time:")
	shiftedprintln(Dates.CompoundPeriod(elapsed_time) |> canonicalize)
	shiftedprintln("ETA:")
	shiftedprintln(Dates.CompoundPeriod(estimated_time) |> canonicalize)
	print("\u001b[2B")
	shiftedprintln("\u001b[32;1mLast Accuracy:\u001b[0m")
	shiftedprintln("\u001b[32m$(accuracy_curve[end])\u001b[0m")
	shiftedprintln("\u001b[34;1mBest Accuracy:\u001b[0m")
	shiftedprintln("\u001b[34m$(best_accuracy)\u001b[0m")
	shiftedprintln("\u001b[35;1mTarget Accuracy:\u001b[0m")
	shiftedprintln("\u001b[35m$(target_accuracy)\u001b[0m")
	print("\r\u001b[$(plot_height - 9)B\n")
end
