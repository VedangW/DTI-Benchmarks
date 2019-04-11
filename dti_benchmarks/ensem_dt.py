from __future__ import division
from __future__ import print_function

import pickle
import random
import numpy as np
import pandas as pd
import concurrent.futures

from time import time
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def load_data_ensemdt(data_path):
	""" 
	Custom load function to load data 
	from data_path, specifically for EnsemDT. 

	Parameters
	----------
	data_path: str
		Path to graph data

	Returns
	-------
	df_pos: pd.DataFrame
		DataFrame with all positive samples
	df_neg: pd.DataFrame
		DataFrame with all negative samples
	df_u: pd.DataFrame
		df of drug feature vectors
	df_v: pd.DataFrame
		df of target feature vectors

	"""

	# Load data
	print ("Loading data from disk...")
	
	with open(data_path) as f:
		graph = pickle.load(f)
	num_u, num_v, u_nodes, v_nodes, y, u_feat, v_feat = graph
	
	print ("Total no. of nodes =", y.shape[0])
	print ("Shape of drug feature tensor =", u_feat.shape)
	print ("Shape of target feature tensor =", v_feat.shape)
	print ("")
	
	# Change to df
	df = np.vstack([u_nodes, v_nodes, y])
	df_transpose = df.T
	df = pd.DataFrame(df_transpose, columns=['u_node', 'v_node', 'y'])
	
	# Separate pos and neg
	df_pos = df[df['y'] == 1]
	df_neg = df[df['y'] == 0]
	
	# De-sparsify feature tensors
	u_feat = u_feat.toarray()
	v_feat = v_feat.toarray()
	
	# Column names for df_u and df_v
	u_feat_headers = ['d' + str(i + 1) for i in range(u_feat.shape[1])]
	v_feat_headers = ['t' + str(i + 1) for i in range(v_feat.shape[1])]
	
	# Create feature dfs
	df_u = pd.DataFrame(u_feat, columns=u_feat_headers)
	df_v = pd.DataFrame(v_feat, columns=v_feat_headers)
	
	print ("Shape of df_u =", df_u.shape)
	print ("Shape of df_v =", df_v.shape)
	print ("Shape of df_pos =", df_pos.shape)
	print ("Shape of df_neg =", df_neg.shape)
	print ("")

	return df_pos, df_neg, df_u, df_v


def train_test_split_ensemdt(df_pos, df_neg, test_size=0.2, shuffle=True):
	"""
	Function to split data into train and test,
	specifically for EnsemDT.

	Parameters
	----------
	df_pos: pd.DataFrame
		See load_data_ensemdt
	df_neg: pd.DataFrame
		See load_data_ensemdt
	test_size: float
		Fraction of training set to set as test set
	shuffle: boolean
		Set to true to shuffle test set


	Returns
	-------
	X_train_pos: pd.DataFrame
		df of positive training samples
	X_train_neg: pd.DataFrame
		df of negative training samples
	X_test: pd.DataFrame
		df of test samples
	y_test: np.array
		Labels for test set
	"""

	# Remove y from pos and neg set
	y_pos = df_pos['y']
	y_neg = df_neg['y']
	
	df_pos_split = df_pos.drop(['y'], axis=1)
	df_neg_split = df_neg.drop(['y'], axis=1)
	
	# Split into pos and neg train and test sets
	X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(df_pos_split, 
													y_pos, test_size=test_size, random_state=42)
	X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(df_neg_split, 
													y_neg, test_size=test_size, random_state=42)
	
	# Recombine to form test set
	X_test_pos['y'] = y_test_pos
	X_test_neg['y'] = y_test_neg
	
	X_test = pd.concat([X_test_pos, X_test_neg])
	
	# Re-enter test labels
	X_train_pos['y'] = y_train_pos
	X_train_neg['y'] = y_train_neg
	
	# Shuffle test set
	if shuffle:
		X_test = X_test.sample(frac=1)
		
	y_test = X_test['y']
	X_test = X_test.drop(['y'], axis=1)
	
	return X_train_pos, X_train_neg, X_test, y_test


class EnsemDT:
	
	def __init__(self, n_estimators=50, dim_red_ratio=0.9, 
				 np_ratio=5, reduce_dims=True, n_components=100,
				 max_depth=None):
		""" 
		Class for EnsemDT model from Ezzat et al. (2017) from
		paper, 'Drug-target interaction prediction using ensemble 
		learning and dimensionality reduction'.

		Parameters
		----------
		n_estimators: int
			No. of base learners
		dim_red_ratio: float
			Parameter 'r'
		np_ratio: float/int
			Ratio of neg set to pos set
		reduce_dims: bool
			Set to true to reduce dimension
		n_components: int
			Number of components after dimensionality
			reduction using PCA
		max_depth: `None` or int
			Max Depth of each decision tree in ensemble
		"""
		
		self.n_estimators = n_estimators
		self.dim_red_ratio = dim_red_ratio
		self.np_ratio = np_ratio
		self.reduce_dims = reduce_dims
		self.n_components = n_components
		self.max_depth = max_depth
		
		self.clfs = list()
		
	def fit(self, df_pos, df_neg, df_u, df_v):
		"""
		Function to train model.

		Parameters
		----------
		df_pos: pd.DataFrame
			DataFrame with all positive samples
		df_neg: pd.DataFrame
			DataFrame with all negative samples
		df_u: pd.DataFrame
			df of drug feature vectors
		df_v: pd.DataFrame
			df of target feature vectors
		"""

		self.num_pos = df_pos.shape[0]
		self.df_u = df_u
		self.df_v = df_v
		
		# Reset training time
		self.training_time = time()
		
		# Loop for each classifier
		for i in tqdm(range(self.n_estimators), desc='Training model...', unit='base learner'):
			# Sample negative instances
			df_neg_sampled = df_neg.sample(self.np_ratio*self.num_pos)
			
			# Concatenate to form training set
			training_set = pd.concat([df_neg_sampled, df_pos])
			training_set = training_set.sample(frac=1)
			
			# Generate column subspaces for drugs and targets
			subspace_u = random.sample(range(self.df_u.shape[1]), 
									   int(self.dim_red_ratio*self.df_u.shape[1]))
			subspace_v = random.sample(range(self.df_v.shape[1]),
									   int(self.dim_red_ratio*self.df_v.shape[1]))

			# Column names for subspaces
			head_u = ['d' + str(i+1) for i in subspace_u]
			head_v = ['t' + str(i+1) for i in subspace_v]
			
			# Retrieve data from subspaces
			df_u_sub = self.df_u[head_u]
			df_v_sub = self.df_v[head_v]
			
			# Dimensionality reduction
			if self.reduce_dims:
				pca_u = PCA(n_components=self.n_components)
				pca_v = PCA(n_components=self.n_components)

				df_u_sub = pca_u.fit_transform(df_u_sub)
				df_v_sub = pca_v.fit_transform(df_v_sub)
			
			# Create final train set
			data = []
			labels = []

			# Loop over rows in training set
			for _, row in training_set.iterrows():
				try:
					# Concatenate drug-target features
					data.append(np.concatenate([df_u_sub[row['u_node']], df_v_sub[row['v_node']]], axis=0))
					# Labels
					labels.append(row['y'])
				except:
					print ("Skipping " + str(row['u_node']) + " " + str(row['v_node']) + "...")
			y = np.vstack(labels)
			y = np.reshape(y, (y.shape[0],))
					
			X = np.vstack(data)
			
			# Fit into classifier
			dt = DecisionTreeClassifier(max_depth=self.max_depth)
			dt.fit(X, y)
			
			# Save metadata
			base_learner = {'clf': dt, 
							'u_cols': head_u, 
							'v_cols': head_v}
			
			# Add tree to classifiers' list
			self.clfs.append(base_learner)
			
		# Update training time
		self.training_time = time() - self.training_time
	
	
	def predict(self, df_test):
		"""
		Function to predict labels for test set.

		Parameters
		----------
		df_test: pd.DataFrame
			df with test instances

		Returns
		-------
		final_preds: np.array
			Array with label predictions
		"""

		# Reset test time
		self.test_time = time()

		preds = list()
		
		# Loop over number of base learners
		for i in tqdm(range(self.n_estimators), desc='Predicting values...', unit='base learner'):
			# Retrieve base learner
			base_learner = self.clfs[i]
			
			# Gather metadata
			head_u = base_learner['u_cols']
			head_v = base_learner['v_cols']
			
			# Retrieve subspaces from data
			df_u_sub = self.df_u[head_u]
			df_v_sub = self.df_v[head_v]
			
			# Dimensionality reduction
			if self.reduce_dims:
				pca_u = PCA(n_components=self.n_components)
				pca_v = PCA(n_components=self.n_components)

				df_u_sub = pca_u.fit_transform(df_u_sub)
				df_v_sub = pca_v.fit_transform(df_v_sub)
			
			# Create final test set
			data = []

			# Loop over rows in test set
			for _, row in df_test.iterrows():
				try:
					# Concatenate drug and target feature vectors
					data.append(np.concatenate([df_u_sub[row['u_node']], df_v_sub[row['v_node']]], axis=0))
				except:
					print ("Skipping " + str(row['u_node']) + " " + str(row['v_node']) + "...")
					
			X_test = np.vstack(data)
			
			# Predict using base learner
			clf = base_learner['clf']
			pred = clf.predict(X_test)

			# Append to list of predictions
			preds.append(pred)
			
		# Average list of predictions
		preds = np.vstack(preds)
		final_preds = np.sum(preds, axis=0).astype(np.float32)
		final_preds /= self.n_estimators
		
		# Update test time
		self.test_time = time() - self.test_time

		return final_preds
	
	
	def predict_proba(self, df_test):
		"""
		Function to estimate probabilities of labels.

		Parameters
		----------
		df_test: pd.DataFrame
			df with test instances

		Returns
		-------
		final_preds: np.array
			Array with probability estimates
		"""
		# Reset test time
		self.test_time = time()

		preds = list()
		
		# Loop over number of base learners
		for i in tqdm(range(self.n_estimators), desc='Predicting values...', unit='base learner'):
			# Retrieve base learner
			base_learner = self.clfs[i]
			
			# Gather metadata
			head_u = base_learner['u_cols']
			head_v = base_learner['v_cols']
			
			# Retrieve subspaces from data
			df_u_sub = self.df_u[head_u]
			df_v_sub = self.df_v[head_v]
			
			# Dimensionality reduction
			if self.reduce_dims:
				pca_u = PCA(n_components=self.n_components)
				pca_v = PCA(n_components=self.n_components)

				df_u_sub = pca_u.fit_transform(df_u_sub)
				df_v_sub = pca_v.fit_transform(df_v_sub)
			
			# Create final test set
			data = []

			# Loop over rows in test set
			for _, row in df_test.iterrows():
				try:
					# Concatenate drug and target feature vectors
					data.append(np.concatenate([df_u_sub[row['u_node']], df_v_sub[row['v_node']]], axis=0))
				except:
					print ("Skipping " + str(row['u_node']) + " " + str(row['v_node']) + "...")
					
			X_test = np.vstack(data)
			
			# Estimate probs using base learner
			clf = base_learner['clf']
			pred = clf.predict_proba(X_test)[:, 1]

			# Append to list of pr. estimates
			preds.append(pred)
			
		# Averate probabilities
		preds = np.vstack(preds)
		final_preds = np.sum(preds, axis=0).astype(np.float32)
		final_preds /= self.n_estimators

		# Update test time
		self.test_time = time() - self.test_time
		
		return final_preds