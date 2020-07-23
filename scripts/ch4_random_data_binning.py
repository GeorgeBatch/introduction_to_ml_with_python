
import numpy as np
import pandas as pd

avg_pos_data = np.random.exponential(1, 1000)
clicks_data = np.abs(np.random.normal(10, 3, 1000))

data = pd.DataFrame({
	'avg_pos': avg_pos_data,
	'clicks': clicks_data
})

min_avg_pos = np.round(np.min(data.avg_pos))
max_avg_pos = np.round(np.max(data.avg_pos))

min_avg_pos = 1
max_avg_pos = 7

num_bins = int((max_avg_pos - min_avg_pos)/0.1)
bins = np.linspace(min_avg_pos, max_avg_pos, num_bins + 1)

digitized = np.digitize(data.avg_pos, bins)
clicks_bin_means = [
	data.clicks[digitized == i].mean() for i in range(1, len(bins))
]
clicks_bin_stds = [
	data.clicks[digitized == i].std() for i in range(1, len(bins))
]
counts_bin = [
	data.clicks[digitized == i].count() for i in range(1, len(bins))
]


data_binned = pd.DataFrame({
	'clicks_bin_means': clicks_bin_means,
	'clicks_bin_stds': clicks_bin_stds,
	'counts_bin': counts_bin
})
