from __future__ import division
from __future__ import print_function

import warnings
import argparse

from ensem_dt import load_data_ensemdt, \
	train_test_split_ensemdt, EnsemDT
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# Create argument parser
parser = argparse.ArgumentParser(description='Test EnsemDT')

parser.add_argument('-d', '--dataset', type=str, 
					default='data_2_small',
                    help='Set path to data')
parser.add_argument('-m', '--max_depth', type=int, default=0,
                    help='Max depth for base learners')

args = parser.parse_args()

# Gather hyperparameters
DATASET = args.path
MAX_DEPTH = args.max_depth

if MAX_DEPTH == 0:
	MAX_DEPTH = None
elif MAX_DEPTH < 0:
	raise ValueError("Max depth must be greater than 0.")

options = ['data_1', 'data_2', 'data_1_small', 'data_2_small']

if DATASET not in options:
	raise ValueError('Invalid dataset.')

if DATASET == 'data_1':
	DATA_PATH = 'data/data_1/data_1.pkl'
elif DATASET == 'data_2':
	DATA_PATH = 'data/data_2/data_2.pkl'
elif DATASET == 'data_2_small':
	DATA_PATH = 'data/data_2_small/graph_2.pkl'
else:
	raise NotImplementedError('Pipeline not implemented.')

# Load data
df_pos, df_neg, df_u, df_v = load_data_ensemdt(DATA_PATH)

# Split into train and test sets
X_train_pos, X_train_neg, X_test, y_test = train_test_split_ensemdt(df_pos, df_neg)

# Initialize and train
e_dt = EnsemDT(max_depth=MAX_DEPTH)
e_dt.fit(X_train_pos, X_train_neg, df_u, df_v)

# Test on test set
pred = e_dt.predict_proba(X_test)
print ("AUROC =", roc_auc_score(y_test, pred))
print ("Total time =", e_dt.training_time + e_dt.test_time, "s")