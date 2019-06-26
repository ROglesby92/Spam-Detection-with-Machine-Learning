# Spam-detection-with-Neural-Networks
Implementing a simple Neural Network to train on data collected from spam emails.

Compare to a K-NN Classification. We first find the best hyperparamters for K-Nearest Neighbore, also check for best distance metric.
We can compare at the end which classification method is more preferable.


The test data is about 4601 lists of data, containing values ranging from all over based of various statistics such as number of words, number of unigram words , longest word, shortest word, etc.

![alt text](https://user-images.githubusercontent.com/33335790/59325683-d6501180-8c98-11e9-8923-dbda661b4974.png)


## To use:  python network.py spambase.data

Can be used with any other dataset that follows the same principles of distribution, real number values for features and binary label result.

Data is split and ready for K-Fold validation.
