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
from sklearn import ensemble, model_selection


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
	data_label = data.pop('medv')

	# test_data preprocessing
	test_data['chas_0'] = 0
	test_data['chas_1'] = 0
	test_data.loc[test_data['chas'] == 0, ['chas_0']] = 1
	test_data.loc[test_data['chas'] == 1, ['chas_1']] = 1
	test_data.drop('chas', inplace=True, axis=1)
	test_id = test_data.pop('ID').tolist()

	# build model and training
	forest = ensemble.RandomForestRegressor()
	parameter_grid = {
		'max_depth': [10, 12, 14, 16, 18, 20, 22, 24],
		'n_estimators': [100, 150, 200, 250, 300, 350, 400]
	}
	grid_search = model_selection.GridSearchCV(estimator=forest, param_grid=parameter_grid, scoring='neg_mean_squared_error')
	grid_search.fit(data, data_label)
	best_params = grid_search.best_params_
	forest = ensemble.RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
										 random_state=False, verbose=False)
	forest.fit(data, data_label)
	# scores = model_selection.cross_val_score(forest, data, data_label, cv=10, scoring='neg_mean_absolute_error')

	# output prediction
	prediction = forest.predict(test_data)
	out_file(prediction, 'boston_housing_competition_random_forest', test_id)


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
