"""
spambase.data FROM
https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data

"""
import csv
import sys
from numpy import *

ERR_FORGIVE = .25
MAX_ITERATIONS = 1000
NUM_SPLITS = 5
ALPHA = .1
file_name = 'spambase.data'
K_VALUE = 3


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


def standardize_data(data):
    for x in data:
        # Get mean.
        _mean = sum(x) / len(x)
        _list = []
        for i in range(len(x)):
            _list.append(pow((i - _mean), 2))

        _std = math.sqrt(sum(_list) / len(_list))

        for i in range(len(x)):
            x[i] = (x[i] - _mean) / _std

def normalize_data(data):
    for x in range(len(data[0])):
        # get min / max
        max = None
        min = None
        for i in range(len(data)):
            if max == None:
                max = data[i][x]
            if min == None:
                min = data[i][x]
            if data[i][x] > max:
                max = data[i][x]
            if data[i][x] < min:
                min = data[i][x]
        # apply the normalization to each value.
        for i in range(len(data)):
            data[i][x] = (data[i][x] - min) / (max - min)


class NN(object):
    def __init__(self, weight_sz, alpha=.25):
        self.learning_rate = alpha
        self.epochs = MAX_ITERATIONS
        # 57 weights for 57 attributes
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

            # create the value to apply
            factor = dot(inputs.T, error * self.sigmoid_derivative(learned))

            # Back propigation
            self.weights += factor * self.learning_rate

    # Forward propigation.
    def learn(self, inputs):
        # sig_derv of Dot product of input and weights
        return self.sigmoid_f(dot(inputs, self.weights))


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


def classify(values, objToClassify, k=1):
    distance = []
    # 0 + 1 Classifications
    for label in values:
        for person in values[label]:  # Go through each person.
            manhattenDistance = 0.0
            l2Distance = 0.0
            for i in range(len(person)):
                #manhattenDistance += abs(person[i] - objToClassify[i])
                l2Distance += (person[i]-objToClassify[i])**2

            # Add a tuple of form (distance,classification) in the distance list
            #distance.append((manhattenDistance, label))
            distance.append((sqrt(l2Distance), label))
    # and select first k distances after sorting by shortest first.
    distance = sorted(distance)[:k]
    freq1 = 0  # frequency of label 0
    freq2 = 0  # frequency of label 1

    for d in distance:
        if d[1] == 0:
            freq1 += 1
        elif d[1] == 1:
            freq2 += 1

    return 0 if freq1 > freq2 else 1

# Creates a dictionary of format {0:[X], 1:[X]}
def create_classifications(data, lables):
    CLASSIFICATIONS = {0: [], 1: []}
    for i in range(len(data)):
        if lables[i] == 0:
            CLASSIFICATIONS[0].append(data[i])
        else:
            CLASSIFICATIONS[1].append(data[i])

    return CLASSIFICATIONS

def test_KNN(data,labels,CL,k):
    correct = 0
    for i in range(len(data)):
        pred = classify(CL, data[i], k)
        if labels[i] == pred:
            correct += 1
    acc = float(correct)/len(data)
    #print("Accuracy: ", acc)
    return acc



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


def main(argv):
    if (len(argv) != 1):
        print('Usage: network.py <DATA> ')
        sys.exit(2)

    # Open and normalize data
    (data, y) = open_file(argv[0])
    normalize_data(data)

    # Splits data into K Folds + 20% Test data
    # trainig_data is 80% test data no folds
    kfolds, training_data = split_data_set(data, y, NUM_SPLITS)

    # First split is 20% of set, take that for test data
    testData = kfolds[0][0][:]
    testLabels = kfolds[0][1][:]

    print "LENGTH OF DATA: " +str(len(kfolds[0][1])*NUM_SPLITS)
    print "NUMBER OF FOLDS: " + str(len(kfolds-1))
    print "FOLD LENGTH: " + str(len(kfolds[0][1]))
    print "NUMBER OF EPOCHS: " + str(MAX_ITERATIONS)
    print "TRAINING RATE: " + str(ALPHA)
    print "ERROR FORGIVENESS: " + str(ERR_FORGIVE)

    best_n = 3
    best_acc = 0.0

    idx = 0
    knn_accuracy_list = []

    # All but training data
    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])
        train_data.remove(fold)
        train_accuracy = []
        for training_fold in train_data:
            train_accuracy.append(test_KNN(training_fold[0],training_fold[1],CL,1))
        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        print "Accuracy of fold: " + str(idx) + ": " + str(foldacc)
        knn_accuracy_list.append(foldacc)
        idx += 1
    best_acc = sum(knn_accuracy_list)/NUM_SPLITS
    print "1NN AVG ACC: = " + str(sum(knn_accuracy_list)/NUM_SPLITS)

    idx = 0
    k3nn_accuracy_list = []

    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])
        train_data.remove(fold)
        train_accuracy = []
        for training_fold in train_data:
            train_accuracy.append(test_KNN(training_fold[0],training_fold[1],CL,3))
        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        print "Accuracy of fold: " + str(idx) + ": " + str(foldacc)
        k3nn_accuracy_list.append(foldacc)
        idx += 1
    if sum(k3nn_accuracy_list)/NUM_SPLITS > best_acc:
        best_acc = sum(k3nn_accuracy_list)/NUM_SPLITS
        best_n = 3
    print "3NN AVG ACC: = " + str(sum(k3nn_accuracy_list)/NUM_SPLITS)

    idx = 0
    k5nn_accuracy_list = []
    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])
        train_data.remove(fold)
        train_accuracy = []
        for training_fold in train_data:
            train_accuracy.append(test_KNN(training_fold[0],training_fold[1],CL,5))
        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        print "Accuracy of fold: " + str(idx) + ": " + str(foldacc)
        k5nn_accuracy_list.append(foldacc)
        idx += 1
    if sum(k5nn_accuracy_list)/NUM_SPLITS > best_acc:
        best_acc = sum(k5nn_accuracy_list)/NUM_SPLITS
        best_n = 5
    print "5NN AVG ACC: = " + str(sum(k5nn_accuracy_list)/NUM_SPLITS)


    print "Testing with best k-nn value: " + str(best_n)
    CL = create_classifications(training_data[0],training_data[1])
    test_ac = test_KNN(testData,testLabels,CL,best_n)
    print "Test ACC: " + str(test_ac)
    print "########\n"
    print "Testing neural_network acc on data...\n"


    # print kfolds[0]
    accuracy_list = []
    # Train each set with each other.
    idx = 0
    for fold in kfolds[1:]:
        # remove test data from set
        train_data = kfolds[1:]
        train_data.remove(fold)
        train_accuracy = []
        for training_fold in train_data:
            # Tuple [([X],[Y])] [0] = tuple [0][0] = data [0][1] = label
            neural_network = NN(weight_sz =len(training_fold[0][0]),alpha=ALPHA)
            inputs = array(training_fold[0], dtype=float128)
            outputs = array([training_fold[1]], dtype=float128).T
            # Trains for MAX_ITERATIONS 100
            neural_network.train(inputs, outputs)
            train_accuracy.append(test_accuracy(fold, neural_network))
        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        print "Accuracy of fold: " + str(idx) + ": " + str(foldacc)
        accuracy_list.append(foldacc)
        idx += 1
    print "AVG TRAIN ACC: = " + str(sum(accuracy_list)/NUM_SPLITS)
    inputs = array(training_data[0], dtype=float128)
    outputs = array([training_data[1]], dtype=float128).T
    neural_network.train(inputs,outputs)
    print "TEST ACC     : = " + str(test_accuracy((testData,testLabels), neural_network))

if __name__ == "__main__":
    main(sys.argv[1:])
# print("Actual awnser: "+str(spam_result) + "|" + "Network confidence: " + str(neural_network.learn(spam_test)))
