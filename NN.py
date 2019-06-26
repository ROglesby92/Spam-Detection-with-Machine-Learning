import csv
import sys
import math
import numpy as np
from numpy import *

# Single Neruon Neural Network.

MAX_ITERATIONS = 1000
ERR_FORGIVE = .5


class NN(object):
    def __init__(self, weight_sz, alpha=.25):
        self.learning_rate = alpha
        self.epochs = MAX_ITERATIONS
        self.weights = 2 * random.random((weight_sz, 1)) - 1

    # Normalize the function to a value between 0 and 1 using sigmoid.
    def sigmoid_f(self, x):
        try:
            return 1. / (1. + exp(-x))
        except OverflowError:
            return 0.

    def sigmoid_derivative(self, x):
        return x * (1. - x)

    def train(self, inputs, outputs):
        for it in xrange(self.epochs):
            # Test with current weights
            learned = self.learn(inputs)

            # Calculate error
            error = outputs - learned
            #if it % 1000 == 0:
            #    print np.mean(1 + error)

            # create the value to apply
            factor = dot(inputs.T, error * self.sigmoid_derivative(learned))

            # Back propigation
            self.weights += factor * self.learning_rate

    # Forward propigation.
    def learn(self, inputs):
        # sig_derv of Dot product of input and weights
        return self.sigmoid_f(dot(inputs, self.weights))


def test_accuracy(test, NN):
    correct = 0
    x = test[0]
    y = test[1]
    for i in range(len(x)):
        activation = NN.learn(x[i])
        if y[i] == 1 and activation >= 1-ERR_FORGIVE:
            correct += 1
        if y[i] == 0 and activation < ERR_FORGIVE:
            correct += 1
    acc = float(correct) / len(x)
    #print("Accuracy: ", acc)
    return acc		


# Split data set into n splits, with 20% test data reserved.
def split_data_set(data, y, n):

    # Get test data, 20% of set
    train_sets = []
    test_len = int(len(data)*.20)
    tData=[]
    tY=[]
    for i in range(test_len):
        r = random.randint(0, len(data))
        tData.append(data.pop(r))
        tY.append(y.pop(r))
    train_sets.append((tData,tY))

    # Make a copy of the rest of the data to train on once we find best hyperparamters

    full_training_set = data[:]
    full_training_labels = y[:]
    fullSetData = []
    fullSetLabels = []
    for i in range(len(data)):
        fullSetData.append(data[i])
        fullSetLabels.append(y[i])

    full_set = (fullSetData,fullSetLabels)


    # Now split the remaining data into N folds
    train_len = int(len(data) / n)

    for k in range(n):
        kdata = []
        ky = []
        for l in range(train_len):
            r = random.randint(0, len(data))
            kdata.append(data.pop(r))
            ky.append(y.pop(r))
        train_sets.append((kdata, ky))
    return train_sets, full_set

	
# Extract the statistics from out data file.
def open_file(filename):
    data = []
    y = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Shuffle data?
        for row in csv_reader:
            # Get spam flag ( determines if spam or not; either 1 or 0 )
            y.append(int(row.pop()))
            # Get the data, convert to floats
            float_list = [float(v) for v in row]
            data.append(float_list)
    return data, y
	

def normalize_data(data):
    for x in range(len(data[0])):
        # get min / max
        _max = None
        _min = None
        for i in range(len(data)):
            if _max == None:
                _max = data[i][x]
            if _min == None:
                _min = data[i][x]
            if data[i][x] > _max:
                _max = data[i][x]
            if data[i][x] < _min:
                _min = data[i][x]
        # apply the normalization to each value.
        for i in range(len(data)):
            try:
                data[i][x] = (data[i][x] - _min) / (_max - _min)
            except ZeroDivisionError:
                data[i][x] = 0	

				
def main(argv):
    if (len(argv) != 1):
	print('Usage: network.py <DATA> ')
	sys.exit(2)
    total_nn_acc = 0.0
    for i in range(10):
	(data, y) = open_file(argv[0])
	normalize_data(data)
	kfolds, training_data = split_data_set(data, y, 5)
	neural_network = NN(weight_sz=len(training_data[0][0]),alpha=.1)
	# First split is 20% of set, take that for test data
        testData = kfolds[0][0][:]
        testLabels = kfolds[0][1][:]    
	print "I : " + str(i)
	inputs = array(training_data[0], dtype=float128)
        outputs = array([training_data[1]], dtype=float128).T
        neural_network.train(inputs,outputs)
        total_nn_acc += test_accuracy((testData,testLabels), neural_network)
    
    print "Final NN Acc: " + str(total_nn_acc/10)
    return

if __name__ == "__main__":
    main(sys.argv[1:])
