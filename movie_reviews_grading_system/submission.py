#!/usr/bin/python

import math
import random
from collections import defaultdict

import util
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    d = defaultdict(int)  # int(),float(),list()...
    lst = x.strip().split()
    for i in range(len(lst)):
        d[lst[i]] += 1
    return d
    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for i in range(numEpochs):
        for j in range(len(trainExamples)):
            feature_vector = featureExtractor(trainExamples[j][0])
            y = 0 if trainExamples[j][1] == -1 else 1
            k = dotProduct(feature_vector, weights)
            h = 1/(1+math.exp(-k))
            increment(weights, -alpha*(h-y), feature_vector)

        # def predictor(x):
        #     return 1 if dotProduct(featureExtractor(x), weights) > 0 else -1
        training_error = evaluatePredictor(trainExamples, lambda x: 1 if dotProduct(featureExtractor(x), weights) > 0 else -1)
        validation_error = evaluatePredictor(validationExamples, lambda x: 1 if dotProduct(featureExtractor(x), weights) > 0 else -1)
        print(f'Training Error: ({i} epoch): {training_error}')
        print(f'Validation Error: ({i} epoch): {validation_error}')
    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrance.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        phi = {}
        length_of_phi = random.randint(1, len(weights))
        for _ in range(length_of_phi):
            key = random.choice(list(weights.keys()))
            phi[key] = phi.get(key, 0)+1
        y = 1 if dotProduct(phi, weights) > 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        new_str = ''
        for i in range(len(x)):
            if x[i] != ' ':
                new_str += x[i]
        # new_str = x.replace(' ', '')
        # new_str = ''.join(x.split())
        extract_dic = {}
        for j in range(len(new_str)-n+1):
            extract_dic[new_str[j:j+n]] = extract_dic.get(new_str[j:j+n], 0)+1
        return extract_dic
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

