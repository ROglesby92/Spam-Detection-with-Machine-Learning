# Spam-detection-with-Neural-Networks
Implementing a simple Neural Network to train on data collected from spam emails.

Compare to a K-NN Classification. We first find the best hyperparamters for K-Nearest Neighbore, also check for best distance metric.
We can compare at the end which classification method is more preferable.


The test data is about 4601 lists of data, containing values ranging from all over based of various statistics such as number of words, number of unigram words , longest word, shortest word, etc.

![alt text](https://user-images.githubusercontent.com/33335790/59325683-d6501180-8c98-11e9-8923-dbda661b4974.png)


## To use:  python network.py spambase.data

or any other dataset that follows the same principles of distribution, real number values for features and binary label result.

Using the outlier removal herustic you will get a result such as this, which tests every metic and parameter:

![alt text](https://user-images.githubusercontent.com/33335790/59325760-11524500-8c99-11e9-9381-7b2400744469.png)


You may specify in the code to perform K-Fold validation or use the Neurel Network.


