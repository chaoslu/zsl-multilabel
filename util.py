import numpy as np


def get_minibatches_idx(n, batch_size, shuffle):
	idx_list = np.arange(n,dtype= np.int32)

	if shuffle:
		np.random.shuffle(idx_list)

	batches_idx = []
	start = 0
	for i in range(n // batch_size):
		batches_idx.append(idx_list[start: start + batch_size])
		start = start + batch_size

	if start != n:
		batches_idx.append(idx_list[start:])

	return batches_idx


def _p(pre,post):
	return "%s_%s" % (pre,post)
