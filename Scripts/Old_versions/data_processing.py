#Made by Rudolfs Berzins 

from __future__ import division
import pandas as pd
import sys
from sys import argv
import re
import numpy as np
from nltk.tokenize import sent_tokenize
import glob
import os
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def open_tsv(tsv_name):
	"""Opens GZiped TSV File Containing the abstracts from MEDLINE

	INPUT: Name of the tsv.gz file

	OUTPUT: Pandas DataFrame of the .tsv file"""
	
	tsv_file = pd.read_table(tsv_name, compression = 'gzip', sep = '\t', names= ["PMID", "Authors", "Journal", "Year", "Title", "Text"])

	return tsv_file

def open_mentions_and_pairs(men_name, pairs_name):
	"""Opens the mentions file and pairs file from tagger made by JensenGroup from NNCPR

	INPUT: mentions file (.tsv), pairs file (.tsv)

	OUTPUT: Pandas DataFrame of the corresponding .tsv files"""

	men_file = pd.read_csv(men_name, sep = '\t', names = ["PMID", "Paragraph", "Sentence", "Char_Start", "Char_End", "Entity_name", "TaxID", "SerialNo"])
	pairs_file = pd.read_csv(pairs_name, sep = '\t', names = ["SerialNo1", "SerialNo2", "Final_Score", "UNUSED1", "UNUSED2", "UNUSED3", "UNUSED4","UNUSED5"])

	return men_file, pairs_file

def open_entities(ents_name):

	ents = pd.read_csv(ents_name, sep = '\t', names = ['Given_ID', 'UNUSED', 'Real_ID'])

	return ents

def open_interactions(interactions_name):
	"""Opens a TSV File Containing the Interactions from STRING DB

	INPUT: Name of the tsv file

	OUTPUT: Pandas DataFrame of the .tsv file"""
	
	interactions_file = pd.read_csv(interactions_name, sep = '\t')

	return interactions_file

def find_common(texts_file, men_file):
	"""Finds the common PubMed ID's between the MEDLINE abstacts and mentions file

	INPUT: Abstracts as PD DataFrame, Mentions as PD DataFrame

	OUTPUT: List of articles (PMID's) both DF's share"""

	journal_list = []
	for i,r in texts_file.iterrows():
		d = re.findall('PMID:\d*', r["PMID"])
		for j in d:
			journal_list.append(int(j[5:]))
	
	set_of_journals = set(men_file["PMID"])
	common = list(set_of_journals & set(journal_list))
	
	return common

def extract_true_pairs(both_entries):
	"""Finds true pairs, meaning those who are in the same paragraph, same sentence, but have different serial ID's and entity names 

	INPUT: Data Frame which consists of entries where both genes are in the same text

	OUTPUT: Data Frame of True Pairs"""

	column_titles = ["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "Entity_name_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "Entity_name_y", "SerialNo_y"]
	results = pd.DataFrame(columns = column_titles)

	row_iterator = both_entries.iterrows()
	_, last = row_iterator.next() #take first item from row_iterator
	for idx,row in row_iterator:
		if row["PMID"] == last["PMID"] and row["Paragraph"] == last["Paragraph"] and row["Sentence"] == last["Sentence"] and row["Entity_name"] != last["Entity_name"] and row["SerialNo"] != last["SerialNo"]:
			
			row_df = pd.DataFrame(row).T #I know this is dirty, but it works
			last_df = pd.DataFrame(last).T #I know this is dirty, but it works
			merger = pd.merge(row_df, last_df, on = ["PMID", "Paragraph", "Sentence", "TaxID"], how = "inner")
			results = results.append(merger, ignore_index = True)
			last = row
		else:
			last = row
			pass

	results[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "SerialNo_y"]] = results[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "SerialNo_y"]].astype(int)
	
	return results

def add_sentences(texts_file, true_pairs_file):
	"""Adds sentences for true pairs 

	INPUT: Texts DF and True Pairs DF

	OUTPUT: True Pairs DF with added sentences"""

	text_copy = texts_file.copy()
	text_copy["Combined"] = None

	#Iterrate over a copy of text file
	for i,r in text_copy.iterrows():
		d = re.findall('PMID:\d*', r["PMID"])
		for j in d:
			if int(j[5:]) in set(true_pairs_file["PMID"].tolist()):
				# Combine Title and Text
				temp_title = r["Title"]
				temp_text = r["Text"]
				text_copy.loc[i,("Combined")] = temp_title + temp_text
				text_of_interest = text_copy.loc[i,("Combined")]

				# Extract Charecter Starts and make them a list
				temp_hits = true_pairs_file.loc[(true_pairs_file['PMID'] == int(j[5:])), ["Char_Start_x"]]

				# Check if list is exactly two entries long
				if len(temp_hits) == 1:
					sublen = 0
					for sentence in sent_tokenize(text_of_interest):
						for ind, line in temp_hits.iterrows():
							if 0 < (line["Char_Start_x"]-37) - sublen < len(sentence): # -37 because that is the difference between len(x) and sys.getsizeof(x)
								true_pairs_file.set_value(ind, 'Text', sentence)
								break
						sublen += len(sentence)

				else:
					pass
	true_pairs_file = true_pairs_file.dropna(axis = 0, how = 'any')
	
	return true_pairs_file

def next_task(all_files):

	if all_files == True:
		return
 
	do_next = raw_input("Do you want to continue? (Yes/No): ")
	if do_next == "No" and all_files == False:
		sys.exit()
	elif do_next != "Yes" and do_next != "No":
		print "TRY AGAIN"
		next_task(all_files)

def produce_pairs(folder, ment, pair):

	print "Producing Pairs data frame"
	#Initialize DF for all results
	column_titles_all = ["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "Entity_name_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "Entity_name_y", "SerialNo_y", "Text"]
	results_all = pd.DataFrame(columns = column_titles_all)
	column_titles_nec = ["Entity_name_x", "SerialNo_x", "Entity_name_y", "SerialNo_y", "Text"]
	results_nec = pd.DataFrame(columns = column_titles_nec)

	#Go over all the files in the specified folder
	for filename in glob.glob(os.path.join(folder,'*tsv.gz')):
		texts = open_tsv(filename)
		mentions, pairs = open_mentions_and_pairs(ment, pair)
		pairs = pairs[["SerialNo1", "SerialNo2", "Final_Score"]].sort_values(['SerialNo1','SerialNo2'])
		mentions = mentions.sort_values('PMID')
		common_j = find_common(texts, mentions)

		shared_entities = mentions[mentions["PMID"].isin(common_j)]	
		shared_entities = shared_entities.sort_values('SerialNo')

		# extract only the common pairs with their respective entries
		ser_no_ents = shared_entities["SerialNo"].tolist()
		temp_s1_values = pairs[pairs["SerialNo1"].isin(ser_no_ents)]
		temp_s2_list = temp_s1_values["SerialNo2"].tolist()
		both_entries = shared_entities[shared_entities["SerialNo"].isin(temp_s2_list)]
		temp_both_list = both_entries["SerialNo"].tolist()
		both_pairs = temp_s1_values[temp_s1_values["SerialNo1"].isin(temp_both_list) & temp_s1_values["SerialNo2"].isin(temp_both_list)]
		both_entries = both_entries.sort_values('PMID')

		#Get True Pairs
		true_pairs = extract_true_pairs(both_entries)

		#Add senteces 
		pairs_with_sentences = add_sentences(texts, true_pairs)
		only_nec = pairs_with_sentences[["Entity_name_x","SerialNo_x", "Entity_name_y", "SerialNo_y", "Text"]]
		

		#Append to results DF
		results_all = results_all.append(pairs_with_sentences, ignore_index = True)
		results_nec = results_nec.append(only_nec, ignore_index = True)
		print "Done with ", filename[54:]

	results_all[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "SerialNo_y"]] = results_all[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "SerialNo_y"]].astype(int)
	results_nec[["SerialNo_x", "SerialNo_y"]] = results_nec[["SerialNo_x", "SerialNo_y"]].astype(int)

	results_all.to_csv('Pairs_With_Sentences_Full.tsv', sep = '\t', index = False, header = False)
	results_nec.to_csv('Pairs_With_Sentences_Only_nec.tsv', sep = '\t', index = False, header = False)
	print "\n", "DONE with Pairs data frame production!", "\n"

	return results_all, results_nec

def produce_pre_bow(pairs_nec, ents):

	print "Producing Pre BOW data frame"

	entities = open_entities(ents)
	pairs_copy = pairs_nec.copy()
	for i,r in pairs_copy.iterrows():

		#Add Real_ID's
		if r["SerialNo_x"] in set(entities["Given_ID"].tolist()) and r["SerialNo_y"] in set(entities["Given_ID"].tolist()):
			x = entities.loc[entities["Given_ID"] == r["SerialNo_x"], ("Real_ID")]
			y = entities.loc[entities["Given_ID"] == r["SerialNo_y"], ("Real_ID")]
			pairs_copy.set_value(i, "Real_ID_x", x.values[0])
			pairs_copy.set_value(i, "Real_ID_y", y.values[0])

		#Mask entity names in text
		try:
			sub1 = re.sub(r["Entity_name_x"], '', r["Text"])
			sub2 = re.sub(r["Entity_name_y"], '', sub1)
		except re.error:
			print "Couldn't mask ", r["Entity_name_y"], " with RegEx\n", "Trying a different way!"
        	sub2 = sub1.replace(r["Entity_name_y"], "")
			pairs_copy.set_value(i, "Text", sub2)

		pairs_copy.set_value(i, "Text", sub2)

	bow_df = pairs_copy[["Real_ID_x", "Real_ID_y", "Text"]] #Get only Real_IDs and Texts
	grouped = bow_df.groupby(["Real_ID_x", "Real_ID_y"]) #Group by Real_ID's
	grouped_text = grouped['Text'].apply(lambda x: ' '.join(x.astype(str))).reset_index() #Concat the text files

	#Initialize some necessary vars
	col_names = ["Real_ID_x", "Real_ID_y", "Text"]
	temp_df = pd.DataFrame(columns = col_names)
	exclude = []

	# Catch semi-duplicate pairs
	for i,r in grouped_text.iterrows():
		pair = [r["Real_ID_x"], r["Real_ID_y"]]
		#Check if there is a reverse version of the pair
		if grouped_text.query('@pair[0] == Real_ID_y and @pair[1] == Real_ID_x').empty: 
			pass
		elif r.name not in exclude: #necessary to not catch the original
			temp = grouped_text.query('@pair[0] == Real_ID_y and @pair[1] == Real_ID_x')
			exclude.append(temp.index[0])
			temp_df = temp_df.append(temp, ignore_index = True)

	#Switch column names
	col_list = list(temp_df)
	col_list[0], col_list[1] = col_list[1], col_list[0]
	temp_df.columns = col_list
	temp_df = temp_df[["Real_ID_x", "Real_ID_y", "Text"]]

	# Append and groupby RealID_s
	new_grouped_text = grouped_text.append(temp_df, ignore_index = True).groupby(["Real_ID_x", "Real_ID_y"])

	#Produce the final DF
	grouped_final = new_grouped_text['Text'].apply(lambda x: ','.join(x.astype(str))).reset_index()

	grouped_final.to_csv('Real_IDs_with_Full_Text.tsv', sep = '\t', index = False, header = False)

	print "\n", "Done with Pre Bow data frame", "\n"

	return grouped_final

def produce_bow_df(pre_bow, ints):

	print "Producing BOW data frame"

	interactions = open_interactions(ints)

	interactions["item_id_a"] = interactions["item_id_a"].str[5:]
	interactions["item_id_b"] = interactions["item_id_b"].str[5:]
	interactions = interactions[["item_id_a", "item_id_b", "mode"]]
	desired_interaction = check_interaction(interactions, 0)

	for i,r in pre_bow.iterrows():
		pre_mode = interactions.loc[interactions["item_id_a"] == r["Real_ID_x"]]
		mode = pre_mode.loc[pre_mode["item_id_b"] == r["Real_ID_y"], ("mode")]
		if mode.empty or desired_interaction not in mode.values:
			pre_bow.set_value(i, "Mode", 0)
		elif desired_interaction in mode.values:
			pre_bow.set_value(i, "Mode", 1)
        
	pre_bow["Mode"] = pre_bow["Mode"].astype(int)
	pre_bow = pre_bow[["Real_ID_x", "Real_ID_y", "Mode", "Text"]]

	pre_bow.to_csv('Bag_of_Words_df.tsv', sep = '\t', index = False)

	print "\n", "Done with Bag of Words data frame", "\n"

	return pre_bow


def check_interaction(interactions, tries = 0):

	desired_interaction = raw_input("What interaction will be positive?: ")
	if desired_interaction in interactions["mode"].tolist():
		return desired_interaction
	elif tries < 10:
		print "There isn't such interaction. Try again"
		tries+=1
		check_interaction(interactions, tries)
	elif tries == 10:
		print "You are hopeless. Please check interaction again! Defaulting to physical binding!"
		desired_interaction = "binding"
		return desired_interaction

def split_train_test(bow_df):

	data = bow_df[["Real_ID_x", "Real_ID_y", "Text"]]
	labels = bow_df["Mode"]

	test_s = input("What is the size of test? (0-1, float): ")
	ran_state = input("Set random state (int): ")

	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_s, random_state=ran_state)
	data_train, data_test, labels_train, labels_test = data_train.reset_index(drop=True), data_test.reset_index(drop=True), labels_train.reset_index(drop=True), labels_test.reset_index(drop=True) 

	data_train.to_csv('Data_train_seed_' + str(ran_state) + '.tsv', sep = '\t', index = False, header = False)
	data_test.to_csv('Data_test_seed_' + str(ran_state) + '.tsv', sep = '\t', index = False, header = False)
	labels_train.to_csv('Labels_train_seed_' + str(ran_state) + '.tsv', sep = '\t', index = False, header = False)
	labels_test.to_csv('Labels_test_seed_' + str(ran_state) + '.tsv', sep = '\t', index = False, header = False)

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

def bag_of_words_and_prediction(bow_df):

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
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(clean_train_texts)

	# Numpy arrays are easy to work with, so convert the result to an 
	# array
	train_data_features = train_data_features.toarray()

	print train_data_features.shape

	print "Training the random forest..."

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 100) 

	# Fit the forest to the training set, using the bag of words as 
	# features and the sentiment labels as the response variable
	#
	# This may take a few minutes to run
	forest = forest.fit( train_data_features, labels_train)

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

	# Use the random forest to make sentiment label predictions
	print "Predicting based on model..."
	result = forest.predict(test_data_features)

	error = get_accuracy(result, labels_test)

	return error

def get_accuracy(l_new, l_te):
	"""Calculates the accuracy of predicted labels, based on the given labels

	INPUT: New(Predicted) Labels, Test Labels

	OUTPUT: Error  """

	acc = 0

	for i in range(len(l_te)):
		if l_new[i] == l_te[i]:
			acc += 1

	acc = float(acc / len(l_te))

	return 1-acc

def main(all_files):
	script, folder, ment, pair, ents, ints = argv
	

	################# PAIRS ###########################
	
	pairs_full, pairs_nec = produce_pairs(folder, ment, pair)
		
	next_task(all_files)
	
	################# END OF PAIRS ###################
	
	################# PRE BOW ########################
	
	pre_bow = produce_pre_bow(pairs_nec, ents) 

	next_task(all_files)

	################# END OF PRE BOW #################
	
	################# BAG OF WORDS DF ################
	
	bow_df = produce_bow_df(pre_bow, ints)

	next_task(all_files)

	################# END OF BAG OF WORDS DF #########

	############ BAG OF WORDS AND RAND FOR ###########
	
	resulting_error = bag_of_words_and_prediction(bow_df)

	print "\n", "The resulting error is ", resulting_error, "\n"
	

if __name__ == "__main__":
	all_files = input("Get all files? (True/False): ")
	main(all_files)
	print "I'm all done!"

	
	

