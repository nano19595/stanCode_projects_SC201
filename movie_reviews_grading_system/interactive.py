"""
File: interactive.py
Name: Allen Li
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
import submission
import util


def main():
	weights = {}
	with open('weights', 'r') as f:
		for line in f:
			weights[line.split()[0]] = float(line.split()[1])
	util.interactivePrompt(submission.extractWordFeatures, weights)


if __name__ == '__main__':
	main()