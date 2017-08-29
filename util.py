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

def my_softmax(x,axis = 0):
	def my_softmax_1d(y):
		diff = y - max(y)
		new_y = diff / sum(np.exp(diff))
		return new_y

	if len(x.shape) == 1:
		new_x = my_softmax_1d(x)
	else:
		if axis == 0:
			new_x = [my_soft_max(x[:,i]) for i in range(x.shape[1])]
		else:
			new_x = [my_soft_max(x[i]) for i in range(x.shape[2])]
	return new_x
