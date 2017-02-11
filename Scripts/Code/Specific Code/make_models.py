from __future__ import division
import pandas as pd
import sys
from sys import argv
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import time

#### TO RUN ####
# python make_models.py Bag_of_Words_df.tsv FeatureCount #
# SLOW AND SHOULD BE RUN ON COMPUTEROME#
###############

def open_bow_df(bow_name):

	bow_df_file = pd.read_csv(bow_name, sep = "\t")

	return bow_df_file

def split_train_test(bow_df):

	data = bow_df[["Real_ID_x", "Real_ID_y", "Text"]]
	labels = bow_df["Mode"]

	test_s = 0.3 #input("What is the size of test? (0-1, float): ")
	ran_state = 1993 #input("Set random state (int): ")

	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_s, random_state=ran_state)
	data_train, data_test, labels_train, labels_test = data_train.reset_index(drop=True), data_test.reset_index(drop=True), labels_train.reset_index(drop=True), labels_test.reset_index(drop=True) 

	return data_train, data_test, labels_train, labels_test

def texts_to_words( raw_text ):
	# Function to convert a raw text to a string of words
	# The input is a single string (a raw text), and 
	# the output is a single string (a preprocessed text)
	#
	# 2. Remove non-letters        
	letters_only = re.sub("[^a-zA-Z0-9]", " ", raw_text) 
	#
	# 3. Convert to lower case, split into individual words
	words = letters_only.lower().split()                             
	#
	# 4. In Python, searching a set is much faster than searching
	#   a list, so convert the stop words to a set
	stops = set(stopwords.words("english"))                  
	# 
	# 5. Remove stop words
	meaningful_words = [w for w in words if not w in stops]   
	#
	# 6. Join the words back into one string separated by space, 
	# and return the result.
	return( " ".join( meaningful_words ))

def bag_of_words_and_prediction(bow_df, feature_count):

	data_train, data_test, labels_train, labels_test = split_train_test(bow_df)

	# Get the number of reviews based on the dataframe column size
	num_texts = data_train["Text"].size

	# Initialize an empty list to hold the clean reviews
	print "Cleaning and parsing the training set article sentences...\n"
	clean_train_texts = []
	for i in xrange( 0, num_texts ):
		# If the index is evenly divisible by 1000, print a message
		# if( (i+1)%100 == 0 ):
			# print "Texts %d of %d\n" % ( i+1, num_texts )                                                                    
		clean_train_texts.append( texts_to_words( data_train["Text"][i] ))

	print "Creating the bag of words...\n"
	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = feature_count) 

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(clean_train_texts)

	# Numpy arrays are easy to work with, so convert the result to an 
	# array
	train_data_features = train_data_features.toarray()

	num_texts = len(data_test["Text"])
	clean_test_texts = [] 

	print "Cleaning and parsing the test set movie reviews...\n"
	for i in xrange(0,num_texts):
		# if( (i+1) % 1000 == 0 ):
			# print "Review %d of %d\n" % (i+1, num_texts)
		clean_texts = texts_to_words( data_test["Text"][i] )
		clean_test_texts.append( clean_texts )

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = vectorizer.transform(clean_test_texts)
	test_data_features = test_data_features.toarray()

	print "Done! Produced train, test split with ", feature_count, " features"

	return train_data_features, test_data_features, labels_train, labels_test

def get_accuracy(l_new, l_te):
	"""Calculates the accuracy of predicted labels, based on the given labels

	INPUT: New(Predicted) Labels, Test Labels

	OUTPUT: Accuracy in percent """

	acc = 0
	for i in range(len(l_te)):
	    if l_new[i] == l_te[i]:
	        acc += 1
	acc = acc/len(l_te)

	return 1-acc

def linear_reg_model(train_features, test_features, labels_train, labels_test):
	from sklearn.linear_model import LinearRegression
	start_time = time.time()

	print "Training Linear Regression model... "
	linear = LinearRegression(normalize = True)

	model = linear.fit(train_features, labels_train)

	print "Predicting based on model... "
	result = model.predict(test_features)

	print "Calculating error..."
	error = get_accuracy(result, labels_test)

	print "The error of linear regression (normalized) ", error

	print "Done"

	finish_time = time.time() - start_time
	print "This took %s seconds" %(finish_time)
	return result, error, finish_time

def logistic_reg_model(train_features, test_features, labels_train, labels_test):
	from sklearn.linear_model import LogisticRegression
	start_time = time.time()

	print "Training Logistic Regression model... "
	logistic = LogisticRegression()

	model = logistic.fit(train_features, labels_train)

	print "Predicting based on model... "
	result = model.predict(test_features)

	print "Calculating error..."
	error = get_accuracy(result, labels_test)

	print "The error of logistic regression ", error

	print "Done"

	finish_time = time.time() - start_time
	print "This took %s seconds" %(finish_time)    
	return result, error, finish_time

def SVM_model(train_features, test_features, labels_train, labels_test):
	from sklearn import svm
	start_time = time.time()
	print "Training Support Vector Machines model... "
	SVM = svm.SVC()

	model = SVM.fit(train_features, labels_train)

	print "Predicting based on model... "
	result = model.predict(test_features)

	print "Calculating error..."
	error = get_accuracy(result, labels_test)

	print "The error of SVM ", error

	print "Done"

	finish_time = time.time() - start_time
	print "This took %s seconds" %(finish_time)
	return result, error, finish_time

def random_forest_model(train_features, test_features, labels_train, labels_test):
	from sklearn.ensemble import RandomForestClassifier
	start_time = time.time()
	print "Training the random forest..."
	forest = RandomForestClassifier(n_estimators = 100, random_state = 23) 

	# Fit the forest to the training set, using the bag of words as 
	# features and the sentiment labels as the response variable
	#
	# This may take a few minutes to run
	forest = forest.fit( train_features, labels_train)

	print "Predicting based on model..."
	result = forest.predict(test_features)

	print "Calculating error..."
	error = get_accuracy(result, labels_test)

	print "The error of random forest with 100 trees is ", error

	print "Done"

	finish_time = time.time() - start_time
	print "This took %s seconds" %(finish_time)
	return result, error, finish_time

def neural_network_model(train_features, test_features, labels_train, labels_test):
	from sklearn.neural_network import MLPClassifier
	start_time = time.time()
	print "Training neural network..."
	clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)

	NN = clf.fit(train_features, labels_train)

	print "Predicting based on model..."
	result = clf.predict(test_features)

	print "Calculating error..."
	error = get_accuracy(result, labels_test)

	print "The error of Neural Network with ", error

	print "Done"

	finish_time = time.time() - start_time
	print "This took %s seconds" %(finish_time)
	return result, error, finish_time

def Gaussian_Naive_Bayes_model(train_features, test_features, labels_train, labels_test):
	from sklearn.naive_bayes import GaussianNB
	start_time = time.time()
	print "Training Gaussian Naive Bayes model... "
	GNB = GaussianNB()

	model = GNB.fit(train_features, labels_train)

	print "Predicting based on model... "
	result = model.predict(test_features)

	print "Calculating error..."
	error = get_accuracy(result, labels_test)

	print "The error of Gaussian Naive Bayes ", error

	print "Done"

	finish_time = time.time() - start_time
	print "This took %s seconds" %(finish_time)
	return result, error, finish_time

def KNN_model(train_features, test_features, labels_train, labels_test):
	from sklearn.neighbors import KNeighborsClassifier
	start_time = time.time()
	print "Training KNN model... "
	KNN = KNeighborsClassifier(n_neighbors = 5)

	model = KNN.fit(train_features, labels_train)

	print "Predicting based on model... "
	result = model.predict(test_features)

	print "Calculating error..."
	error = get_accuracy(result, labels_test)

	print "The error of KNN ", error

	print "Done"

	finish_time = time.time() - start_time
	print "This took %s seconds" %(finish_time)
	return result, error, finish_time

def main():

	_, bow_df, number = argv
	bag_of_words_df = open_bow_df(bow_df)
	col_names = ["Calssification Model", "Feature Count", "Error", "Time"]
	model_df = pd.DataFrame(columns = col_names)

	train_data_features, test_data_features, labels_train, labels_test = bag_of_words_and_prediction(bag_of_words_df, int(number))
	_, ran_for_error_1000, ran_for_time_1000 = random_forest_model(train_data_features, test_data_features, labels_train, labels_test)
	_, NN_error_1000, NN_time_1000 = neural_network_model(train_data_features, test_data_features, labels_train, labels_test)
	_, linear_error_1000, linear_time_1000 = linear_reg_model(train_data_features, test_data_features, labels_train, labels_test)
	_, logistic_error_1000, logistic_time_1000 = logistic_reg_model(train_data_features, test_data_features, labels_train, labels_test)
	_, SVM_error_1000, SVM_time_1000 = SVM_model(train_data_features, test_data_features, labels_train, labels_test)
	_, GNB_error_1000, GNB_time_1000 = Gaussian_Naive_Bayes_model(train_data_features, test_data_features, labels_train, labels_test)
	_, KNN_error_1000, KNN_time_1000 = KNN_model(train_data_features, test_data_features, labels_train, labels_test)
	model_df.loc[0] = ["Random Forest (100 trees, random state = 23)", int(number), ran_for_error_1000, ran_for_time_1000]
	model_df.loc[1] = ["Neural Network", int(number), NN_error_1000, NN_time_1000]
	model_df.loc[2] = ["Linear Regression", int(number), linear_error_1000, linear_time_1000]
	model_df.loc[3] = ["Logistic Regression", int(number), logistic_error_1000, logistic_time_1000]
	model_df.loc[4] = ["Support Vector Machines", int(number), SVM_error_1000, SVM_time_1000]
	model_df.loc[5] = ["Gaussian Naive Bayes", int(number), GNB_error_1000, GNB_time_1000]
	model_df.loc[6] = ["K Nearest Neighbor (5)", int(number), KNN_error_1000, KNN_time_1000]
	model_df.to_csv("Model_df_"+number+".tsv", sep = '\t', index = False, header = False)
	del train_data_features, test_data_features, labels_train, labels_test

if __name__ == "__main__":
	main()