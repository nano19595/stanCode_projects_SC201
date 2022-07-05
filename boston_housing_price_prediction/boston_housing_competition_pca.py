"""
File: boston_housing_competition.py
Name: 
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientist!
"""

import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, decomposition


TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'


def main():
	data = pd.read_csv(TRAIN_FILE)
	test_data = pd.read_csv(TEST_FILE)

	# data preprocessing
	data.drop('ID', inplace=True, axis=1)
	data['chas_0'] = 0
	data['chas_1'] = 0
	data.loc[data['chas'] == 0, ['chas_0']] = 1
	data.loc[data['chas'] == 1, ['chas_1']] = 1
	data.drop('chas', inplace=True, axis=1)

	# test_data preprocessing
	test_data['chas_0'] = 0
	test_data['chas_1'] = 0
	test_data.loc[test_data['chas'] == 0, ['chas_0']] = 1
	test_data.loc[test_data['chas'] == 1, ['chas_1']] = 1
	test_data.drop('chas', inplace=True, axis=1)
	test_id = test_data.pop('ID').tolist()

	# split training and validation data
	train_data, val_data = model_selection.train_test_split(data, test_size=0.1)
	train_data_label = train_data.pop('medv')
	val_data_label = val_data.pop('medv')

	# standardize all data
	standardizer = preprocessing.StandardScaler()
	train_data = standardizer.fit_transform(train_data)
	val_data = standardizer.transform(val_data)
	test_data = standardizer.transform(test_data)

	# conduct PCA
	pca = decomposition.PCA(n_components=8)
	train_data = pca.fit_transform(train_data)
	# var_retained = sum(pca.explained_variance_ratio_)
	# print(f'var_retained:{var_retained}')
	val_data = pca.transform(val_data)
	test_data = pca.transform(test_data)

	# generate poly features
	poly = preprocessing.PolynomialFeatures(degree=2)
	train_data = poly.fit_transform(train_data)
	val_data = poly.transform(val_data)
	test_data = poly.transform(test_data)

	# build model and training
	h = linear_model.Ridge(alpha=2)
	classifier = h.fit(train_data, train_data_label)
	training_acc = classifier.score(train_data, train_data_label)
	print(f'training_acc:{training_acc}')

	# validating
	val_acc = classifier.score(val_data, val_data_label)
	print(f'val_acc:{val_acc}')

	# output prediction
	prediction = classifier.predict(test_data)
	out_file(prediction, 'boston_housing_competition_1', test_id)


def out_file(predictions, filename, test_id):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for i in range(len(predictions)):
			out.write(str(test_id[i]) + ',' + str(predictions[i]) + '\n')
	print('===============================================')


if __name__ == '__main__':
	main()
