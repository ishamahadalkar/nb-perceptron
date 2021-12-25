# nb-perceptron
Implementing the Naive Bayes and Perceptron algorithms from scratch in Python

training filenames: correspond to a subset of the Titanic data (in the same format as train-file.data
and train-file.label) that are used as the training set in the algorithm.
test filenames: correspond to another subset of the Titanic data (in the same format as test-file.data
and test-file.label) that are used as the test set in the algorithm.

1. Naive Bayes: nbc.py

Input: 
python nbc.py train-file.data train-file.label test-file.data test-file.label

Sample Output:
ZERO-ONE LOSS=0.XXXX
SQUARED LOSS=0.YYYY Test Accuracy=0.YYYY

2. Perceptron: perceptron.py

Input:
python perceptron.py train-file.data train-file.label test-file.data test-file.label

Output:
Hinge LOSS=0.ZZZZ
Test Accuracy=0.YYYY

3. Average Perceptron: average.py

Input:
python average.py train-file.data train-file.label test-file.data test-file.label

Output:
Hinge LOSS=0.ZZZZ
Test Accuracy=0.YYYY