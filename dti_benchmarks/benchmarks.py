from __future__ import print_function
from __future__ import division

import pickle
import argparse
import numpy as np
import scipy.sparse as sp

import warnings
warnings.filterwarnings('ignore')

from pipeline import load_data_from_disk

from time import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Associated names of classifiers
classifier_names = {'dt': 'DecisionTreeClassifier',
					'rf': 'RandomForestClassifier',
					'lr': 'LogisticRegression',
					'svc': 'SupportVectorClassifier'}

def run_benchmark(classifier):
	""" Utility function to run the a classifier

		Parameters
		----------
		clf_data: Eg. ('dt', DecisionTreeClassifier())
	"""

	clf_name, clf = classifier
	info_str = "Using " + classifier_names[clf_name] + "."
	print (info_str)

	if CV:
		run_classifier_cross_val(clf)
	else:
		run_classifier(clf)


def run_classifier(clf):
	""" Run with classifier to test a single time

		Parameters
		----------
		clf: sklearn classifier
	"""
	t0 = time()
	clf.fit(X_train, y_train)
	pred = clf.predict_proba(X_test)[:, 1]
	auc = roc_auc_score(y_test, pred)

	print ('AUROC achieved =', auc)
	print ('Time taken =', time() - t0, 's')
	print ('')


def run_classifier_cross_val(clf, k=5):
	""" Run with classifier for k-fold cross-validation.
		
		Parameters
		----------
		clf: sklearn classifier
		k: number of cv folds
	"""
	t0 = time()
	scores = cross_val_score(clf, X, y, cv=k, scoring='roc_auc')

	print ('Scores =', scores)
	print ('Mean AUROC achieved =', np.mean(scores))
	print ('Time taken =', time() - t0, 's')
	print ('')


# Argument parser
parser = argparse.ArgumentParser(description='Benchmarks')
parser.add_argument('-cv', '--cross_validate', action='store_true',
                    help='Set to use cross validation.')
parser.add_argument('-d', '--dataset', type=str, default='data_2_small',
                    help='Name of dataset to use.')

args = parser.parse_args()

# Hyperparameters
CV = args.cross_validate
DATASET = args.dataset

if CV:
	print ('Using cross validation.')

print ('Using dataset:', DATASET)

# Load data from disk and split
X, y = load_data_from_disk(dataset=DATASET)
X_train, X_test, y_train, y_test = \
	train_test_split(X, y, stratify=y,
			test_size=0.2, shuffle=True)

print ('Shape of X_train =', X_train.shape)
print ('Shape of y_train =', y_train.shape)
print ('Shape of X_test =', X_test.shape)
print ('Shape of y_test =', y_test.shape)
print ('')

print ('Running benchmarks...')
print ('')

# List of classifiers
classifiers = list()
# classifiers.append(('dt', DecisionTreeClassifier()))
classifiers.append(('rf', RandomForestClassifier()))
classifiers.append(('lr', LogisticRegression()))
# classifiers.append(('svc', SVC()))

# Run benchmarks
for classifier in classifiers:
	run_benchmark(classifier)

print ('Done.')