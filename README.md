# Spam-detection-with-Neural-Networks

Implementing a single layer Neural Network and K Nearest Neighbore algorithim in Python.

I test multiple distance metrics such as: Manhatten Distance, L2 Distance and Cosine Similarity
to find the best hyperparameters for K-NN and compare results to a Neural Network in terms of speed and accuracy.


The test data is about 4601 lists of data, containing values ranging from all over based of various statistics such as number of words, number of unigram words , longest word, shortest word, etc.

![alt text](https://user-images.githubusercontent.com/33335790/59325683-d6501180-8c98-11e9-8923-dbda661b4974.png)


## To use:  python KNN.py spambase.data  |   python NN.py spambase.data

Can be used with any other dataset that follows the same principles of distribution, real number values for features and binary label result.

Data is split and ready for K-Fold validation.
