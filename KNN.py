import csv
import sys
import math
import numpy as np
from numpy import *
import time
# K-Nearest Neighbor Python implmentation (NO NUMPY)
# Evalutes 3 Distance metrics: Manhatten, L2 and Cosine Similarity distance.

"""
classify: K-NN implementation.

Args:
K   : K-NN distance metric. Best values are typically 1,3,5,10,15 dependending on dataset
measure: Either L2, cosign or Manhantten distance(default)
objToClassisy: Row of attributes that is to be classified.
values: Data that K-NN compares neighbores from.

"""
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

  
# Helper function, tests each distance metric with passed K argument and data/labels.
# Returns accuracy of each metric.
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
  
  
# Helper Function for classify.
# Splits data into two parts, (0)non-classified and (1)classified
# Creates a dictionary of format {0:[X], 1:[X]}
def create_classifications(data, lables):
    CLASSIFICATIONS = {0: [], 1: []}
    for i in range(len(data)):
        if lables[i] == 0:
            CLASSIFICATIONS[0].append(data[i])
        else:
            CLASSIFICATIONS[1].append(data[i])

    return CLASSIFICATIONS  
  
# Compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
def cosine_similarity(v1,v2):
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
        print('Usage: KNN.py <DATASET> ')
	      sys.exit(2)
    
    # How many test iterations? (Remember, it will take longer for bigger dimensional datasets (15+ features))
    num_epochs = 5
    
    # Avg test values for each metric and K value(1,3,5)
    man_avg_knn1 = 0.0
    man_avg_knn3 = 0.0
    man_avg_knn5 = 0.0
    
    l2_avg_knn1 = 0.0
    l2_avg_knn3 = 0.0
    l2_avg_knn5 = 0.0
    
    cosign_avg_knn1 = 0.0
    cosign_avg_knn3 = 0.0
    cosign_avg_knn5 = 0.0
    
    
    for i in range(num_epochs):
        # trainig_data is 80% test data no folds
        kfolds, training_data = split_data_set(data, y, NUM_SPLITS)
        # First split is 20% of set, take that for test data
        testData = kfolds[0][0][:]
        testLabels = kfolds[0][1][:]
        CL = create_classifications(training_data[0],training_data[1])
        
        # Test K values 1,3,5 
        # Add values to each metrics total.
        x,y,z = test_KNN(testData,testLabels,CL,1)
        man_avg_knn1 += x
        l2_avg_knn1 += y
        cosign_avg_knn1 += z
        x,y,z = test_KNN(testData,testLabels,CL,3)
        man_avg_knn3 += x
        l2_avg_knn3 += y
        cosign_avg_knn3 += z
        x,y,z = test_KNN(testData,testLabels,CL,5)
        man_avg_knn5 += x
        l2_avg_knn5 += y
        cosign_avg_knn5 += z

    print "FINAL TOTAL ACC REPORT . . ."
    
    print "MANHATTEN 1-NN: " + str(man_avg_knn1 / num_epochs)
    print "MANHATTEN 3-NN: " + str(man_avg_knn3 / num_epochs)
    print "MANHATTEN 5-NN: " + str(man_avg_knn5 / num_epochs)

    print "------------------------------"
    print "L2 1-NN: " + str(l2_avg_knn1 / num_epochs)
    print "L2 3-NN: " + str(l2_avg_knn3 / num_epochs)
    print "L2 5-NN: " + str(l2_avg_knn5 / num_epochs)

    print "------------------------------"
    print "Cosign 1-NN: " + str(cosign_avg_knn1 / num_epochs)
    print "Cosign 3-NN: " + str(cosign_avg_knn3 /num_epochs)
    print "Cosign 5-NN: " + str(cosign_avg_knn5 / num_epochs)

    print "------------------------------"

if __name__ == "__main__":
    main(sys.argv[1:])
