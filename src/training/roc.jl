"""
    roc(scores, answers)

Compute Receiver operating characteristic (ROC) curve. Based on [scikit
implementation](github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/metrics/_ranking.py)
to allow the comparison of systems created in both platforms.
"""
function roc(scores, answers)

	sorted_indices = sortperm(scores, rev = true)

	scores = scores[sorted_indices]
	answers = answers[sorted_indices]

	# Scores may have repeated points. Those are removed to prevent the curve
	# going straight up or right, as in these values both TPR and FPR grow
	unique_threshold_idxs = [findall(x -> x != 0, diff(scores)); length(scores)]

	# The true positive ratio can be evaluated by dividing the cumulative sum
	# of positive answers at each threshold by the total sum of positive answers.
	tp_sum = cumsum(answers)[unique_threshold_idxs]
	fp_sum = unique_threshold_idxs .- tp_sum
	return (fp_sum/fp_sum[end], tp_sum/tp_sum[end], scores[unique_threshold_idxs])

end


"""
    auc(x, y)

Evaluate the area under the given curve using the trapezoidal rule.
"""
auc(x, y) = sum(diff(x) .* (y[1:end-1] .+ y[2:end]) ./ 2)

