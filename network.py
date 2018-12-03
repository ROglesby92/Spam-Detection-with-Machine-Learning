"""
spambase.data FROM 
https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data

"""
import csv
from numpy import *

data = []
herustic = []

# Extract the statistics from out data file.

with open('spambase.data') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		# Get spam flag ( determines if spam or not; either 1 or 0 )
		herustic.append(int(row.pop()))
		# Get the data, convert to floats
		float_list = [float(v) for v in row]
		data.append(float_list)
    
	#normalize the data between 1 and 0.
	#for each attribute in each training set data
  for x in range(len(data[0])):
		# get min / max 
		max = 0.0
		min = 1.0
		for i in range(len(data)):
			if data[i][x] > max:
				max = data[i][x]
			if data[i][x] < min:
				min = data[i][x]
		# apply the normalization to each value.
		for i in range(len(data)):
			data[i][x] = (data[i][x] - min) / (max - min)
			
		
	
print ("Length of training set :" + str(len(data)))
print ("Number of characteristics of each training data :" + str(len(data[0])))
	
		
class NN(object):
	def __init__(self):
		# Sets random to start from same seed everytime, so testing will be more consistent.
		# random.seed(1)

		# 57 random weights to start off with.
		self.weights = 2 * random.random((57,1))-1
		
	# Normalize the function to a value between 0 and 1 using sigmoid.
	def sigmoid_f(self,x):
		return 1 / (1+exp(-x))

	def sigmoid_derivative(self,x):
		return x * (1-x)


	def train(self,inputs,outputs,training_iterations):
		for it in xrange(training_iterations):

			# Test with current weights
			learned = self.learn(inputs)

			# Calculate error
			error = outputs - learned

			# create the value to apply
			factor = dot(inputs.T, error * self.sigmoid_derivative(learned))
			
			# Back propigation
			self.weights += factor
		

	# Forward propigation.
	def learn(self,inputs):
		# sig_derv of Dot product of input and weights
		return self.sigmoid_f(dot(inputs,self.weights))



#Initialize 
neural_network = NN()

# Get our values from our data set.
    
inputs = array(data,dtype=float128);

outputs = array([herustic],dtype=float128).T;

training_cycles = 300

neural_network.train(inputs, outputs, training_cycles);

print("Training  {} CYCLES..".format(training_cycles));

# Grab 10 random samples
for x in range(10):
  r = random.randint(1,4601);
  
  spam_test = inputs[r]
  spam_result = outputs[r];
  
  print("Actual awnser: "+str(spam_result) + "|" + "Network confidence: " + str(neural_network.learn(spam_test)));

 
 
  
	
