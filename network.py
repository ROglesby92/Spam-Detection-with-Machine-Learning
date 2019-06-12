import csv
import sys
import math
import numpy as np
from numpy import *
import time


ERR_FORGIVE = .5
MAX_ITERATIONS = 10000
NUM_SPLITS = 5
ALPHA = .1

MAX_HERUISTIC = 0
MIN_HERUISTIC = 0


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
            try:
                data[i][x] = (data[i][x] - min) / (max - min)
            except ZeroDivisionError:
                data[i][x] = 0



# Find iregularitys inside the data set to improve accuracy.
def find_outliers(data):
    threshold=3
    outliers=[]
    attributes = [[] for i in range(len(data[0]))]
    numremoved = 0
    # Get all attributes of each test sample.
    for i in range(len(data)):
        for j in range(len(data[i])):
            attributes[j].append(data[i][j])

    for k in range(len(attributes)):
        mean_1 = np.mean(attributes[k])
        std_1 =np.std(attributes[k])

        for y in attributes[k]:
            if std_1 == 0:
                z_score = 0
            else:
                z_score = (y - mean_1)/std_1


            if np.abs(z_score) > threshold:
                # remove this test set from the entire data set
                # outliers.append(y)
                # set outlier to zero. will try to remove next.
                #y = 0
                y = mean_1
                numremoved += 1

    #print "Removed " + str(numremoved) + " "
#return outliers
# Finds MOST influencel attribute towards a YES(1) CLASSIFICATION
# Finds LEAST influencel attribute towards a NO(0) CLASSIFICATION
# return elements that are not weighted one way or another, even elements.
def generate_hueristic(data,labels,numElements=20):
    w = [ 0.0 for i in range(len(data[0])) ]

    for i in range(len(data)):
        for j in range(len(data[i])):
            if labels[i] == 0:  # No classificaiton
                w[j] -= data[i][j]
            else:               # Yes classification
                w[j] += data[i][j]


    hList = set()
    threshhold = 1
    targetAmount = len(w) - numElements
    # Trim list elements until there is 20
    while(len(hList) < targetAmount):
        for z in w:
            if z> -threshhold and z<threshhold and (len(hList) < targetAmount):
                #hList.append(w.index(z))
                hList.add(w.index(z))
        threshhold += 1
        # incremenet threshhold to get next layer of elements

    return hList

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

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    try:
        return sumxy/math.sqrt(sumxx*sumyy)
    except ZeroDivisionError:
        return 0.

def classify(values, objToClassify, k=1,measure="L2"):
    distance = []
    # 0 + 1 Classifications
    for label in values:
        for person in values[label]:  # Go through each person.
            manhattenDistance = 0.0
            l2Distance = 0.0
            cosignsimDistance = 0.0
            if measure == "cosign":
                cosignsimDistance += cosine_similarity(person,objToClassify)
                distance.append((-cosignsimDistance, label))
            else:
                for i in range(len(person)):
                    if measure == "L2":
                            l2Distance += (person[i]-objToClassify[i])**2
                    else:
                            manhattenDistance += abs(person[i] - objToClassify[i])
                        # Add a tuple of form (distance,classification) in the distance list

                if measure == "L2":
                        distance.append((sqrt(l2Distance), label))
                else:
                    distance.append((manhattenDistance, label))

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
    correct2 = 0
    correct3 = 0
    for i in range(len(data)):
        pred = classify(CL, data[i], k,measure="Manhatten")
        pred2 = classify(CL, data[i], k,measure="L2")
        pred3 = classify(CL, data[i], k,measure="cosign")
        if labels[i] == pred:
            correct += 1
        if labels[i] == pred2:
            correct2 += 1
        if labels[i] == pred3:
            correct3 += 1
    acc = float(correct)/len(data)
    acc2 = float(correct2)/len(data)
    acc3 = float(correct3)/len(data)

    #print("Accuracy: ", acc)
    return acc,acc2,acc3


def remove_attributes(datalist,hlist):
    idx = 0
    for i in hlist:
        j = i - idx
        for row in datalist:
            row.pop(j)
        idx += 1

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

    do_nn_only = False
    # Open and normalize data
    #(data, y) = open_file(argv[0])
    #normalize_data(data)
    #find_outliers(data)
    #kfolds, training_data = split_data_set(data, y, NUM_SPLITS)

    if do_nn_only:
        total_nn_acc = 0.0
        for i in range(10):
            (data, y) = open_file(argv[0])
            normalize_data(data)
            find_outliers(data)
            kfolds, training_data = split_data_set(data, y, NUM_SPLITS)
            #t0 = time.time()
            neural_network = NN(weight_sz=len(training_data[0][0]),alpha=ALPHA)
            # First split is 20% of set, take that for test data
            testData = kfolds[0][0][:]
            testLabels = kfolds[0][1][:]
            print "I : " + str(i)
            #print "NUMBER OF EPOCHS: " + str(MAX_ITERATIONS)
            #print "TRAINING RATE: " + str(ALPHA)
            #print "ERROR FORGIVENESS: " + str(ERR_FORGIVE)
            inputs = array(training_data[0], dtype=float128)
            outputs = array([training_data[1]], dtype=float128).T
            neural_network.train(inputs,outputs)
            #t1 = time.time()
            #print "TEST ACC: " + str(test_accuracy((testData,testLabels), neural_network))
            total_nn_acc += test_accuracy((testData,testLabels), neural_network)
            #total = t1-t0
            #print "TOTAL TRAIN + TEST TIME: " + str(total)
            #return
        print "Final NN Acc: " + str(total_nn_acc/10)
        return


    # Attributes that are equally distributed among spam and regular mail
    #h = generate_hueristic(data,y,10.0)
    #remove_attributes(data,h)

    man_avg_knn1 = 0.0
    man_avg_knn3 = 0.0
    man_avg_knn5 = 0.0
    man_avg_knn10 = 0.0
    man_avg_knn15 = 0.0

    l2_avg_knn1 = 0.0
    l2_avg_knn3 = 0.0
    l2_avg_knn5 = 0.0
    l2_avg_knn10 = 0.0
    l2_avg_knn15 = 0.0

    cosign_avg_knn1 = 0.0
    cosign_avg_knn3 = 0.0
    cosign_avg_knn5 = 0.0
    cosign_avg_knn10 = 0.0
    cosign_avg_knn15 = 0.0



    num_epochs = 10
    remove_outliers = False
    do_herustic = True
    do_kfold_validation = False

    if remove_outliers:
        print "Removing outliers..."
    if do_herustic:
        print "Reducing number of attributes via herustic"

    for i in range(num_epochs):
        if do_kfold_validation:
            break
        print "ITER: " + str(i)
        (data, y) = open_file(argv[0])

        normalize_data(data)
        if remove_outliers:
            find_outliers(data)

        if do_herustic:
            h = generate_hueristic(data,y)
            remove_attributes(data,h)

        # Attributes that are equally distributed among spam and regular mail
        #h = generate_hueristic(data,y)
        #remove_attributes(data,h)

        # Splits data into K Folds + 20% Test data
        # trainig_data is 80% test data no folds
        kfolds, training_data = split_data_set(data, y, NUM_SPLITS)

        # First split is 20% of set, take that for test data
        testData = kfolds[0][0][:]
        testLabels = kfolds[0][1][:]


        #print "Starting on regular test data. . .:\n"
        #print "NUMBER OF ATTRIBUTES REMOVED FROM HERUISTIC: " + str(len(h))

        #print "\n1-NN . . ."
        t_0 = time.time()
        CL = create_classifications(training_data[0],training_data[1])
        x,y,z = test_KNN(testData,testLabels,CL,1)
        t_1 = time.time()
        _total = t_1 - t_0
        print "Total time train 1-NN: " +str(_total/3.0)
        #print "Manhatten Test ACC: " + str(x)
        #print "L2 Test ACC: " + str(y)
        #print "Cosign Test ACC: " + str(z)
        #print "------------------------------"

        man_avg_knn1 += x
        l2_avg_knn1 += y
        cosign_avg_knn1 += z


        x,y,z = test_KNN(testData,testLabels,CL,3)

        man_avg_knn3 += x
        l2_avg_knn3 += y
        cosign_avg_knn3 += z

        #print "\n3-NN"
        #print "Manhatten Test ACC: " + str(x)
        #print "L2 Test ACC: " + str(y)
        #print "Cosign Test ACC: " + str(z)
        #print "------------------------------"
        x,y,z = test_KNN(testData,testLabels,CL,5)

        man_avg_knn5 += x
        l2_avg_knn5 += y
        cosign_avg_knn5 += z

        x,y,z = test_KNN(testData,testLabels,CL,10)

        man_avg_knn10 += x
        l2_avg_knn10 += y
        cosign_avg_knn10 += z

        x,y,z = test_KNN(testData,testLabels,CL,15)

        man_avg_knn15 += x
        l2_avg_knn15 += y
        cosign_avg_knn15 += z

        #print "\n5-NN"
        #print "Manhatten Test ACC: " + str(x)
        #print "L2 Test ACC: " + str(y)
        #print "Cosign Test ACC: " + str(z)
        #print "------------------------------"

    print "FINAL TOTAL ACC REPORT . . ."

    print "MANHATTEN 1-NN: " + str(man_avg_knn1 / num_epochs)
    print "MANHATTEN 3-NN: " + str(man_avg_knn3 / num_epochs)
    print "MANHATTEN 5-NN: " + str(man_avg_knn5 / num_epochs)
    print "MANHATTEN 10-NN: " + str(man_avg_knn10 / num_epochs)
    print "MANHATTEN 15-NN: " + str(man_avg_knn15 / num_epochs)
    print "------------------------------"
    print "L2 1-NN: " + str(l2_avg_knn1 / num_epochs)
    print "L2 3-NN: " + str(l2_avg_knn3 / num_epochs)
    print "L2 5-NN: " + str(l2_avg_knn5 / num_epochs)
    print "L2 10-NN: " + str(l2_avg_knn10 / num_epochs)
    print "L2 15-NN: " + str(l2_avg_knn15 / num_epochs)
    print "------------------------------"
    print "Cosign 1-NN: " + str(cosign_avg_knn1 / num_epochs)
    print "Cosign 3-NN: " + str(cosign_avg_knn3 /num_epochs)
    print "Cosign 5-NN: " + str(cosign_avg_knn5 / num_epochs)
    print "Cosign 10-NN: " + str(cosign_avg_knn10 / num_epochs)
    print "Cosign 15-NN: " + str(cosign_avg_knn15 / num_epochs)
    print "------------------------------"




    # For evaluating K-FOLD Splits and determing best K hyperparameter for K-NN


    if not do_kfold_validation:
        return

    print "LENGTH OF DATA: " +str(len(kfolds[0][1])*NUM_SPLITS)
    print "NUMBER OF FOLDS: " + str(len(kfolds)-1)
    print "FOLD LENGTH: " + str(len(kfolds[0][1]))




    best_n = 1
    best_n2 = 1
    best_n3 = 1
    best_acc = 0.0
    best_acc2 = 0.0
    best_acc3 = 0.0

    final_acc = 0.0

    idx = 0
    knn_accuracy_list = []
    knn_accuracy_list2 = []
    knn_accuracy_list3 = []

    # All but training data

    print "\nEvaluating 1-KK Nearest Neighbor..."
    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])

        train_data.remove(fold)
        train_accuracy = []
        train_accuracy2 = []
        train_accuracy3 = []
        for training_fold in train_data:
            x,y,z = test_KNN(training_fold[0],training_fold[1],CL,1)
            train_accuracy.append(x)
            train_accuracy2.append(y)
            train_accuracy3.append(z)

        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        foldacc2 = sum(train_accuracy2)/(NUM_SPLITS-1.0)
        foldacc3 = sum(train_accuracy3)/(NUM_SPLITS-1.0)

        #print "[Fold "+ str(idx+1)+"-"+str(int(float(idx+1)/NUM_SPLITS*100))+"%]" + "Manhatten "+ ": " + str(foldacc) + "|" + " L2 "+ ": " + str(foldacc2) + "|" + " Cosign "+ ": " + str(foldacc3)

        knn_accuracy_list.append(foldacc)
        knn_accuracy_list2.append(foldacc2)
        knn_accuracy_list3.append(foldacc3)
        idx += 1

    best_acc = sum(knn_accuracy_list)/NUM_SPLITS
    best_acc2 = sum(knn_accuracy_list2)/NUM_SPLITS
    best_acc3 = sum(knn_accuracy_list3)/NUM_SPLITS


    dist_avg_acc = (best_acc+best_acc2+best_acc3)/3

    print "\nManhatten 1-NN   AVG ACC: " + str(sum(knn_accuracy_list)/NUM_SPLITS)
    print "L2 1-NN          AVG ACC: " + str(sum(knn_accuracy_list2)/NUM_SPLITS)
    print "Cosine 1-NN      AVG ACC: " + str(sum(knn_accuracy_list3)/NUM_SPLITS)
    print "\n1-NN Total AVG distance ACC: " + str(dist_avg_acc)
    idx = 0
    k3nn_accuracy_list = []
    k3nn_accuracy_list2 = []
    k3nn_accuracy_list3 = []

    print "\nEvaluating 3-KK Nearest Neighbor..."
    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])
        train_data.remove(fold)
        train_accuracy = []
        train_accuracy2 = []
        train_accuracy3 = []
        for training_fold in train_data:
            x,y,z = test_KNN(training_fold[0],training_fold[1],CL,3)
            train_accuracy.append(x)
            train_accuracy2.append(y)
            train_accuracy3.append(z)

        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        foldacc2 = sum(train_accuracy2)/(NUM_SPLITS-1.0)
        foldacc3 = sum(train_accuracy3)/(NUM_SPLITS-1.0)

        #print "[Fold "+ str(idx+1)+"-"+str(int(float(idx+1)/NUM_SPLITS*100))+"%]" + "Manhatten "+ ": " + str(foldacc) + "|" + " L2 "+ ": " + str(foldacc2) + "|" + " Cosign "+ ": " + str(foldacc3)


        k3nn_accuracy_list.append(foldacc)
        k3nn_accuracy_list2.append(foldacc2)
        k3nn_accuracy_list3.append(foldacc3)
        idx += 1

    if sum(k3nn_accuracy_list)/NUM_SPLITS > best_acc:
        best_acc = sum(k3nn_accuracy_list)/NUM_SPLITS
        best_n = 3
    if sum(k3nn_accuracy_list2)/NUM_SPLITS > best_acc2:
        best_acc2 = sum(k3nn_accuracy_list2)/NUM_SPLITS
        best_n2 = 3
    if sum(k3nn_accuracy_list3)/NUM_SPLITS > best_acc3:
        best_acc3 = sum(k3nn_accuracy_list3)/NUM_SPLITS
        best_n3 = 3

    dist_avg_acc2 = (best_acc+best_acc2+best_acc3)/3

    print "\nManhatten 3-NN   AVG ACC: " + str(sum(k3nn_accuracy_list)/NUM_SPLITS)
    print "L2 3-NN          AVG ACC: " + str(sum(k3nn_accuracy_list2)/NUM_SPLITS)
    print "Cosine 3-NN      AVG ACC: " + str(sum(k3nn_accuracy_list3)/NUM_SPLITS)
    print "\n3-NN Total AVG distance ACC: " + str(dist_avg_acc2)


    idx = 0
    k5nn_accuracy_list = []
    k5nn_accuracy_list2 = []
    k5nn_accuracy_list3 = []

    print "\nEvaluating 5-KK Nearest Neighbor..."
    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])
        train_data.remove(fold)
        train_accuracy = []
        train_accuracy2 = []
        train_accuracy3 = []

        for training_fold in train_data:
            x,y,z = test_KNN(training_fold[0],training_fold[1],CL,5)
            train_accuracy.append(x)
            train_accuracy2.append(y)
            train_accuracy3.append(z)

        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        foldacc2 = sum(train_accuracy2)/(NUM_SPLITS-1.0)
        foldacc3 = sum(train_accuracy3)/(NUM_SPLITS-1.0)

        #print "[Fold "+ str(idx+1)+"-"+str(int(float(idx+1)/NUM_SPLITS*100))+"%]" + "Manhatten "+ ": " + str(foldacc) + "|" + " L2 "+ ": " + str(foldacc2) + "|" + " Cosign "+ ": " + str(foldacc3)

        k5nn_accuracy_list.append(foldacc)
        k5nn_accuracy_list2.append(foldacc2)
        k5nn_accuracy_list3.append(foldacc3)
        idx += 1

    if sum(k5nn_accuracy_list)/NUM_SPLITS > best_acc:
        best_acc = sum(k5nn_accuracy_list)/NUM_SPLITS
        best_n = 5
    if sum(k5nn_accuracy_list2)/NUM_SPLITS > best_acc2:
        best_acc2 = sum(k5nn_accuracy_list2)/NUM_SPLITS
        best_n2 = 5
    if sum(k5nn_accuracy_list3)/NUM_SPLITS > best_acc3:
        best_acc3 = sum(k5nn_accuracy_list3)/NUM_SPLITS
        best_n3 = 5

    dist_avg_acc3 = (best_acc+best_acc2+best_acc3)/3
    print "\nManhatten 5-NN   AVG ACC: " + str(sum(k5nn_accuracy_list)/NUM_SPLITS)
    print "L2 5-NN          AVG ACC: " + str(sum(k5nn_accuracy_list2)/NUM_SPLITS)
    print "Cosine 5-NN      AVG ACC: " + str(sum(k5nn_accuracy_list3)/NUM_SPLITS)
    print "\n5-NN Total AVG distance ACC: " + str(dist_avg_acc3)

    idx = 0
    k10nn_accuracy_list = []
    k10nn_accuracy_list2 = []
    k10nn_accuracy_list3 = []

    print "\nEvaluating 10-KK Nearest Neighbor..."
    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])
        train_data.remove(fold)
        train_accuracy = []
        train_accuracy2 = []
        train_accuracy3 = []

        for training_fold in train_data:
            x,y,z = test_KNN(training_fold[0],training_fold[1],CL,5)
            train_accuracy.append(x)
            train_accuracy2.append(y)
            train_accuracy3.append(z)

        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        foldacc2 = sum(train_accuracy2)/(NUM_SPLITS-1.0)
        foldacc3 = sum(train_accuracy3)/(NUM_SPLITS-1.0)

        #print "[Fold "+ str(idx+1)+"-"+str(int(float(idx+1)/NUM_SPLITS*100))+"%]" + "Manhatten "+ ": " + str(foldacc) + "|" + " L2 "+ ": " + str(foldacc2) + "|" + " Cosign "+ ": " + str(foldacc3)

        k10nn_accuracy_list.append(foldacc)
        k10nn_accuracy_list2.append(foldacc2)
        k10nn_accuracy_list3.append(foldacc3)
        idx += 1

    if sum(k10nn_accuracy_list)/NUM_SPLITS > best_acc:
        best_acc = sum(k10nn_accuracy_list)/NUM_SPLITS
        best_n = 10
    if sum(k10nn_accuracy_list2)/NUM_SPLITS > best_acc2:
        best_acc2 = sum(k10nn_accuracy_list2)/NUM_SPLITS
        best_n2 = 10
    if sum(k10nn_accuracy_list3)/NUM_SPLITS > best_acc3:
        best_acc3 = sum(k10nn_accuracy_list3)/NUM_SPLITS
        best_n3 = 10

    dist_avg_acc4 = (best_acc+best_acc2+best_acc3)/3

    print "\nManhatten 10-NN   AVG ACC: " + str(sum(k10nn_accuracy_list)/NUM_SPLITS)
    print "L2 10-NN          AVG ACC: " + str(sum(k10nn_accuracy_list2)/NUM_SPLITS)
    print "Cosine 10-NN     AVG ACC: " + str(sum(k10nn_accuracy_list3)/NUM_SPLITS)
    print "\n10-NN Total AVG distance ACC: " + str(dist_avg_acc4)


    idx = 0
    k15nn_accuracy_list = []
    k15nn_accuracy_list2 = []
    k15nn_accuracy_list3 = []
    print "\nEvaluating 15-KK Nearest Neighbor..."
    for fold in kfolds[1:]:
        #break # Remove if testing KNN
        train_data = kfolds[1:]
        CL = create_classifications(fold[0], fold[1])
        train_data.remove(fold)
        train_accuracy = []
        train_accuracy2 = []
        train_accuracy3 = []

        for training_fold in train_data:
            x,y,z = test_KNN(training_fold[0],training_fold[1],CL,5)
            train_accuracy.append(x)
            train_accuracy2.append(y)
            train_accuracy3.append(z)

        foldacc = sum(train_accuracy)/(NUM_SPLITS-1.0)
        foldacc2 = sum(train_accuracy2)/(NUM_SPLITS-1.0)
        foldacc3 = sum(train_accuracy3)/(NUM_SPLITS-1.0)

        #print "[Fold "+ str(idx+1)+"-"+str(int(float(idx+1)/NUM_SPLITS*100))+"%]" + "Manhatten "+ ": " + str(foldacc) + "|" + " L2 "+ ": " + str(foldacc2) + "|" + " Cosign "+ ": " + str(foldacc3)

        k15nn_accuracy_list.append(foldacc)
        k15nn_accuracy_list2.append(foldacc2)
        k15nn_accuracy_list3.append(foldacc3)
        idx += 1

    if sum(k15nn_accuracy_list)/NUM_SPLITS > best_acc:
        best_acc = sum(k15nn_accuracy_list)/NUM_SPLITS
        best_n = 15
    if sum(k15nn_accuracy_list2)/NUM_SPLITS > best_acc2:
        best_acc2 = sum(k15nn_accuracy_list2)/NUM_SPLITS
        best_n2 = 15
    if sum(k15nn_accuracy_list3)/NUM_SPLITS > best_acc3:
        best_acc3 = sum(k15nn_accuracy_list3)/NUM_SPLITS
        best_n3 = 15

    dist_avg_acc5 = (best_acc+best_acc2+best_acc3)/3

    print "\nManhatten 15-NN   AVG ACC: " + str(sum(k15nn_accuracy_list)/NUM_SPLITS)
    print "L2 15-NN          AVG ACC: " + str(sum(k15nn_accuracy_list2)/NUM_SPLITS)
    print "Cosine 15-NN     AVG ACC: " + str(sum(k15nn_accuracy_list3)/NUM_SPLITS)
    print "\n15-NN Total AVG distance ACC: " + str(dist_avg_acc5)



    #accList = [dist_avg_acc,dist_avg_acc2,dist_avg_acc3]
    #best_acc = max(dist_avg_acc,dist_avg_acc2,dist_avg_acc3)
    #best_idx = accList.index(best_acc)
    #best_found_n = 0
    #if best_idx == 0:
    #    best_found_n = 1
    #if best_idx == 1:
    #    best_found_n = 3
    #if best_idx == 2:
    #    best_found_n = 5

    #print "\nTesting with best found K-NN value: " + str(best_found_n)
    return
    print "Testing with on test data now..:\n"
    CL = create_classifications(training_data[0],training_data[1])
    x,y,z = test_KNN(testData,testLabels,CL,1)

    print "1-NN"
    print "Manhatten Test ACC: " + str(x)
    print "L2 Test ACC: " + str(y)
    print "Cosign Test ACC: " + str(z)

    x,y,z = test_KNN(testData,testLabels,CL,3)

    print "\n3-NN"
    print "Manhatten Test ACC: " + str(x)
    print "L2 Test ACC: " + str(y)
    print "Cosign Test ACC: " + str(z)

    x,y,z = test_KNN(testData,testLabels,CL,5)

    print "\n5-NN"
    print "Manhatten Test ACC: " + str(x)
    print "L2 Test ACC: " + str(y)
    print "Cosign Test ACC: " + str(z)


    print "\nTesting neural_network acc on data...\n"


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
        print "["+str(int(float(idx+1)/NUM_SPLITS*100))+"%] " +  "ACC FOLD " + str(idx+1) + ": " + str(foldacc)
        accuracy_list.append(foldacc)
        idx += 1
    print "AVG TRAIN ACC: = " + str(sum(accuracy_list)/NUM_SPLITS)
    inputs = array(training_data[0], dtype=float128)
    outputs = array([training_data[1]], dtype=float128).T
    neural_network.train(inputs,outputs)
    print "TEST ACC     : " + str(test_accuracy((testData,testLabels), neural_network))

if __name__ == "__main__":
    main(sys.argv[1:])
# print("Actual awnser: "+str(spam_result) + "|" + "Network confidence: " + str(neural_network.learn(spam_test)))

