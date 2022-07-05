"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
import util
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	first_row = False
	with open(filename, 'r') as f:
		for line in f:
			if not first_row:
				first_row = True
				col_lst = line.strip().split(',')
				for i in range(len(col_lst)):
					if mode == 'Train':
						if i not in [0, 3, 8, 10]:
							data[col_lst[i]] = []
					else:
						if i not in [0, 2, 7, 9]:
							data[col_lst[i]] = []
			else:
				personal_data_lst = line.strip().split(',')
				if mode == 'Train':
					start_index = 2
					if personal_data_lst[start_index+4] != '' and personal_data_lst[start_index+10] != '':
						choose_line = True
						if personal_data_lst[1] == '0':
							data['Survived'].append(0)
						elif personal_data_lst[1] == '1':
							data['Survived'].append(1)
				else:
					start_index = 1
					choose_line = True
				if choose_line:
					for j in range(len(personal_data_lst)):
						if j == start_index:
							if personal_data_lst[j] == '1':
								data['Pclass'].append(1)
							elif personal_data_lst[j] == '2':
								data['Pclass'].append(2)
							elif personal_data_lst[j] == '3':
								data['Pclass'].append(3)
						elif j == start_index+3:
							if personal_data_lst[j] == 'male':
								data['Sex'].append(1)
							elif personal_data_lst[j] == 'female':
								data['Sex'].append(0)
						elif j == start_index+4:
							if personal_data_lst[j] != '':
								data['Age'].append(float(personal_data_lst[j]))
							else:
								mean_age = round(sum(training_data['Age'])/len(training_data['Age']), 3)
								data['Age'].append(mean_age)
						elif j == start_index+5:
							data['SibSp'].append(int(personal_data_lst[j]))
						elif j == start_index+6:
							data['Parch'].append(int(personal_data_lst[j]))
						elif j == start_index+8:
							if personal_data_lst[j] != '':
								data['Fare'].append(float(personal_data_lst[j]))
							else:
								mean_fare = round(sum(training_data['Fare']) / len(training_data['Fare']), 3)
								data['Fare'].append(mean_fare)
						elif j == start_index+10:
							if personal_data_lst[j] == 'S':
								data['Embarked'].append(0)
							elif personal_data_lst[j] == 'C':
								data['Embarked'].append(1)
							elif personal_data_lst[j] == 'Q':
								data['Embarked'].append(2)
					choose_line = False
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	unique_ele_num = len(set(data[feature]))
	for i in range(unique_ele_num):
		data[f'{feature}_{i}'] = []
	for ele in data[feature]:
		if feature == 'Sex':
			if ele == 0:
				data['Sex_0'].append(1)
				data['Sex_1'].append(0)
			else:
				data['Sex_1'].append(1)
				data['Sex_0'].append(0)

		elif feature == 'Pclass':
			if ele == 1:
				data['Pclass_0'].append(1)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(0)
			elif ele == 2:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(1)
				data['Pclass_2'].append(0)
			else:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(1)

		elif feature == 'Embarked':
			if ele == 0:
				data['Embarked_0'].append(1)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(0)
			elif ele == 1:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(1)
				data['Embarked_2'].append(0)
			elif ele == 2:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(1)
	data.pop(feature)
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	############################
	for key in data.keys():
		min_ = min(data[key])
		max_ = max(data[key])
		for i in range(len(data[key])):
			data[key][i] = (data[key][i]-min_)/(max_-min_)
	############################
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# Step 2 : Feature Extract
	feature_extract = {}
	# Step 3 : Start training
	for _ in range(num_epochs):
		for i in range(len(labels)):
			if degree == 1:
				for key in keys:
					feature_extract[key] = inputs[key][i]
			elif degree == 2:
				for key in keys:
					feature_extract[key] = inputs[key][i]
				for j in range(len(keys)):
					for k in range(j, len(keys)):
						feature_extract[keys[j] + keys[k]] = inputs[keys[j]][i] * inputs[keys[k]][i]
			y = labels[i]
			k = util.dotProduct(feature_extract, weights)
			h = 1/(1+math.exp(-k))
			# Step 4 : Update weights
			util.increment(weights, -alpha*(h-y), feature_extract)
	return weights
