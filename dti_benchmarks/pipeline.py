from __future__ import print_function
from __future__ import division

import pickle
import numpy as np
import scipy.sparse as sp

def load_data_from_disk(dataset='data_2_small'):
	""" Options for dataset:
		'data_1'
		'data_2'
		'data_1_small'
		'data_2_small'
	"""
	options = ['data_1', 'data_2', 'data_1_small', 'data_2_small']

	if dataset not in options:
		raise ValueError('Invalid dataset.')

	if dataset == 'data_2_small':
		print('Loading data_2_small from disk...')

		with open('data/graph_2.pkl') as f:
		    graph = pickle.load(f)
		num_u, num_v, u_nodes, v_nodes, y, u_feat, v_feat = graph

		u_feat = u_feat.toarray()
		v_feat = v_feat.toarray()

		X = list()
		for i in range(len(u_nodes)):
		    x = np.concatenate([u_feat[u_nodes[i]], v_feat[v_nodes[i]]], axis=0)
		    X.append(x)
		X = np.vstack(X)

		print ('Done.')

	else:
		raise NotImplementedError('Pipeline not implemented yet.')

	return X, y



