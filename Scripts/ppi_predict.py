"""Made by Rudolfs Berzins
Febuary 2017
v1.0

Framework for predicting protein-protein interactions"""

from __future__ import division
import re
import glob
import os
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


class PPIclassifier(object):
    """Has Three Functions:
    1. setup - just loads dataframes for use in other functions
    2. produce_dfs - Produce Various Pandas DataFrames, needs two args:
    a number for random state and proportion of test_set compared to the full set (default:0.3)
    3. predict - .... """
    def setup(self, files):
        """Opens files provided in a list with the following order -
        0 - Path to folder containing MEDLINE abstracts,
        1 - Hits file from Text-mining tagger,
        2 - Pairs file from Text-mining tagger,
        3 - ORGANISM entities file from ORGANISM dictionary,
        4 - Protein interactions file for ORGANISM from STRING"""
        if type(files) != list:
            raise ValueError("You need to provide a list of filenames in the following order specified by docstring")
        self.ments, self.pairs = open_mentions_and_pairs(files[1], files[2])
        self.entities = open_entities(files[3])
        self.interactions = open_interactions(files[4])
        self.files = files

        return self.ments, self.pairs, self.entities, self.interactions

    def produce_dfs(self, ran_state, test_prop=0.3):
        """Produces Various Pandas DataFrames:
        * Pairs With Sentences (Both Full and Necessary)
        * Pre_BagOfWords DataFrame
        * Real BagOfWords DataFrame
        * Train and Test Sets"""
        self.pairs_DF = produce_pairs(self.files[0], self.ments, self.pairs)
        self.pre_BOW = produce_pre_bow(self.pairs_DF, self.entities)
        self.BOW = produce_bow_df(self.pre_BOW, self.interactions)
        self.train_set, self.test_set = manual_trte_split(self.BOW, ran_state, test_prop)

        return self.train_set, self.test_set

    def predict(self, n_feat, n_trees, random_state,
                train=None, test=None):
        """Predict Results using Random Forests"""
        if train is not None and test is not None:
            train = train
            test = test
        else:
            train = self.train_set
            test = self.test_set

        self.train_data = train[["Real_ID_x", "Real_ID_y", "Text"]]
        self.train_labels = train["Mode"].tolist()
        self.test_data = test[["Real_ID_x", "Real_ID_y", "Text"]]
        self.test_labels = test["Mode"].tolist()

        self.train_feat_vec, self.test_feat_vec = bag_of_words_feat_vecs(self.train_data,
                                                                         self.test_data,
                                                                         n_feat)
        self.error, self.probs = RF_classifier(self.train_feat_vec,
                                               self.test_feat_vec,
                                               self.train_labels,
                                               self.test_labels,
                                               n_trees,
                                               random_state)
        return self.error, self.probs





#### Functions


def open_tsv(tsv_name):
    """Opens GZiped TSV File Containing the abstracts from MEDLINE

    INPUT: Name of the tsv.gz file

    OUTPUT: Pandas DataFrame of the .tsv file"""

    tsv_file = pd.read_table(tsv_name,
                             compression='gzip',
                             sep='\t',
                             names=["PMID", "Authors", "Journal", "Year", "Title", "Text"])

    return tsv_file

def open_mentions_and_pairs(men_name, pairs_name):
    """Opens the mentions file and pairs file from tagger made by JensenGroup from NNCPR

    INPUT: mentions file (.tsv), pairs file (.tsv)

    OUTPUT: Pandas DataFrame of the corresponding .tsv files"""

    men_file = pd.read_csv(men_name,
                           sep='\t',
                           names=["PMID", "Paragraph", "Sentence", "Char_Start",
                                  "Char_End", "Entity_name", "TaxID", "SerialNo"])
    pairs_file = pd.read_csv(pairs_name,
                             sep='\t',
                             names=["SerialNo1", "SerialNo2", "Final_Score",
                                    "UNUSED1", "UNUSED2", "UNUSED3", "UNUSED4",
                                    "UNUSED5"])

    return men_file, pairs_file

def open_entities(ents_name):
    """Opens the mentions file and pairs file from tagger made by JensenGroup from NNCPR

    INPUT: mentions file (.tsv), pairs file (.tsv)

    OUTPUT: Pandas DataFrame of the corresponding .tsv files"""

    ents = pd.read_csv(ents_name,
                       sep='\t',
                       names=['Given_ID', 'UNUSED', 'Real_ID'])

    return ents

def open_interactions(interactions_name):
    """Opens a TSV File Containing the Interactions from STRING DB

    INPUT: Name of the tsv file

    OUTPUT: Pandas DataFrame of the .tsv file"""

    interactions_file = pd.read_csv(interactions_name, sep='\t')

    return interactions_file

def find_common(texts_file, men_file):
    """Finds the common PubMed ID's between the MEDLINE abstacts and mentions file

    INPUT: Abstracts as PD DataFrame, Mentions as PD DataFrame

    OUTPUT: List of articles (PMID's) both DF's share"""

    journal_list = []
    for _, rows in texts_file.iterrows():
        pmid = re.findall('PMID:\d*', rows["PMID"])
        for j in pmid:
            journal_list.append(int(j[5:]))

    set_of_journals = set(men_file["PMID"])
    common = list(set_of_journals & set(journal_list))

    return common

def extract_true_pairs(both_entries):
    """Finds true pairs, meaning those who are in the same paragraph, same sentence,
    but have different serial ID's and entity names

    INPUT: Data Frame which consists of entries where both genes are in the same text

    OUTPUT: Data Frame of True Pairs"""

    column_titles = ["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "Entity_name_x",
                     "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "Entity_name_y",
                     "SerialNo_y"]
    results = pd.DataFrame(columns=column_titles)

    row_iterator = both_entries.iterrows()
    _, last = row_iterator.next() #take first item from row_iterator
    for _, row in row_iterator:
        if row["PMID"] == last["PMID"] and row["Paragraph"] == last["Paragraph"] and row["Sentence"] == last["Sentence"] and row["Entity_name"] != last["Entity_name"] and row["SerialNo"] != last["SerialNo"]:

            row_df = pd.DataFrame(row).T #I know this is dirty, but it works
            last_df = pd.DataFrame(last).T #I know this is dirty, but it works
            merger = pd.merge(row_df, last_df, on=["PMID", "Paragraph", "Sentence", "TaxID"],
                              how="inner")
            results = results.append(merger, ignore_index=True)
            last = row
        else:
            last = row

    results[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID", "SerialNo_x",
             "Char_Start_y", "Char_End_y", "SerialNo_y"]] = results[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "SerialNo_y"]].astype(int)
    return results

def add_sentences(texts_file, true_pairs_file):
    """Adds sentences for true pairs

    INPUT: Texts DF and True Pairs DF

    OUTPUT: True Pairs DF with added sentences"""

    text_copy = texts_file.copy()
    text_copy["Combined"] = None

    #Iterrate over a copy of text file
    for idx, row in text_copy.iterrows():
        pmid = re.findall('PMID:\d*', row["PMID"])
        for j in pmid:
            if int(j[5:]) in set(true_pairs_file["PMID"].tolist()):
                # Combine Title and Text
                temp_title = row["Title"]
                temp_text = row["Text"]
                text_copy.loc[idx, ("Combined")] = temp_title + temp_text
                text_of_interest = text_copy.loc[idx, ("Combined")]

                # Extract Charecter Starts and make them a list
                temp_hits = true_pairs_file.loc[(true_pairs_file['PMID'] == int(j[5:])),
                                                ["Char_Start_x"]]

                # Check if list is exactly two entries long
                if len(temp_hits) == 1:
                    sublen = 0
                    for sentence in sent_tokenize(text_of_interest):
                        for ind, line in temp_hits.iterrows():
                            if 0 < (line["Char_Start_x"]-37) - sublen < len(sentence):
                            # -37 because that is the difference between len(x) and sys.getsizeof(x)
                                true_pairs_file.set_value(ind, 'Text', sentence)
                                break
                        sublen += len(sentence)

                else:
                    pass
    true_pairs_file = true_pairs_file.dropna(axis=0, how='any')

    return true_pairs_file

def produce_pairs(folder, ment, pair):
    """Make Pairs with Sentences DataFrame which includes pairs paired with sentences.
    Will produce 2 DataFrames:
    Pairs with Sentences Full (Includes all textual information)
    Pairs with sentences Necessary (Includes only what is needed further)

    Will only return the PwS Necessary DataFrame"""

    print "Producing Pairs data frame"
    #Initialize DF for all results
    column_titles_all = ["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x",
                         "Entity_name_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y",
                         "Entity_name_y", "SerialNo_y", "Text"]
    results_all = pd.DataFrame(columns=column_titles_all)
    column_titles_nec = ["Entity_name_x", "SerialNo_x", "Entity_name_y", "SerialNo_y", "Text"]
    results_nec = pd.DataFrame(columns=column_titles_nec)

    #Go over all the files in the specified folder
    for filename in glob.glob(os.path.join(folder, '*tsv.gz')):
        try:
            texts = open_tsv(filename)
            pair = pair[["SerialNo1",
                         "SerialNo2",
                         "Final_Score"]].sort_values(['SerialNo1', 'SerialNo2'])
            ment = ment.sort_values('PMID')
            common_j = find_common(texts, ment)

            shared_entities = ment[ment["PMID"].isin(common_j)]
            shared_entities = shared_entities.sort_values('SerialNo')

            # extract only the common pairs with their respective entries
            ser_no_ents = shared_entities["SerialNo"].tolist()
            temp_s1_values = pair[pair["SerialNo1"].isin(ser_no_ents)]
            temp_s2_list = temp_s1_values["SerialNo2"].tolist()
            both_entries = shared_entities[shared_entities["SerialNo"].isin(temp_s2_list)]
            # temp_both_list = both_entries["SerialNo"].tolist()
            # both_pairs = temp_s1_values[temp_s1_values["SerialNo1"].isin(temp_both_list) &
            #                             temp_s1_values["SerialNo2"].isin(temp_both_list)]
            both_entries = both_entries.sort_values('PMID')

            #Get True Pairs
            true_pairs = extract_true_pairs(both_entries)

            #Add senteces

            pairs_with_sentences = add_sentences(texts, true_pairs)
            only_nec = pairs_with_sentences[["Entity_name_x", "SerialNo_x", "Entity_name_y",
                                             "SerialNo_y", "Text"]]

            results_all = results_all.append(pairs_with_sentences, ignore_index=True)
            results_nec = results_nec.append(only_nec, ignore_index=True)
            print "Done with ", filename[54:]

        except (KeyError, UnicodeDecodeError, StopIteration):

            print "Something went wrong with ", filename[54:], "\n", "Ignoring it!"

        #Append to results DF


    results_all[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID",
                 "SerialNo_x", "Char_Start_y", "Char_End_y", "SerialNo_y"]] = results_all[["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y", "SerialNo_y"]].astype(int)
    results_nec[["SerialNo_x", "SerialNo_y"]] = results_nec[["SerialNo_x", "SerialNo_y"]].astype(int)

    results_all.to_csv('Pairs_With_Sentences_Full.tsv', sep='\t', index=False, header=False)
    results_nec.to_csv('Pairs_With_Sentences_Only_nec.tsv', sep='\t', index=False, header=False)
    print "\n", "DONE with Pairs data frame production!", "\n"

    return results_nec

def produce_pre_bow(pairs_nec, ents):
    """Adding this later"""

    print "Producing Pre BOW data frame"

    pairs_copy = pairs_nec.copy()
    for idx, row in pairs_copy.iterrows():

        #Add Real_ID's
        if row["SerialNo_x"] in set(ents["Given_ID"].tolist()) and row["SerialNo_y"] in set(ents["Given_ID"].tolist()):
            first = ents.loc[ents["Given_ID"] == row["SerialNo_x"], ("Real_ID")]
            second = ents.loc[ents["Given_ID"] == row["SerialNo_y"], ("Real_ID")]
            pairs_copy.set_value(idx, "Real_ID_x", first.values[0])
            pairs_copy.set_value(idx, "Real_ID_y", second.values[0])

        #Mask entity names in text
        try:
            sub1 = re.sub(row["Entity_name_x"], '', row["Text"])
            sub2 = re.sub(row["Entity_name_y"], '', sub1)
        except re.error:
            print "Couldn't mask ", row["Entity_name_y"], " with RegEx\n", "Trying a different way"
            sub2 = sub1.replace(row["Entity_name_y"], "")
            pairs_copy.set_value(idx, "Text", sub2)

        pairs_copy.set_value(idx, "Text", sub2)

    bow_df = pairs_copy[["Real_ID_x", "Real_ID_y", "Text"]] #Get only Real_IDs and Texts
    #Produce the final DF
    grouped_final = bow_df

    grouped_final.to_csv('Real_IDs_with_Full_Text.tsv', sep='\t', index=False, header=False)

    print "\n", "Done with Pre Bow data frame", "\n"

    return grouped_final

def produce_bow_df(pre_bow, ints):
    """Adding later"""

    print "Producing BOW data frame"

    interactions = ints

    interactions["item_id_a"] = interactions["item_id_a"].str[5:]
    interactions["item_id_b"] = interactions["item_id_b"].str[5:]
    interactions = interactions[["item_id_a", "item_id_b", "mode"]]
    # desired_interaction = check_interaction(interactions, 0)

    desired_interaction = "binding"

    for idx, row in pre_bow.iterrows():
        pre_mode = interactions.loc[interactions["item_id_a"] == row["Real_ID_x"]]
        mode = pre_mode.loc[pre_mode["item_id_b"] == row["Real_ID_y"], ("mode")]
        if mode.empty or desired_interaction not in mode.values:
            pre_bow.set_value(idx, "Mode", 0)
        elif desired_interaction in mode.values:
            pre_bow.set_value(idx, "Mode", 1)

    pre_bow["Mode"] = pre_bow["Mode"].astype(int)
    pre_bow = pre_bow[["Real_ID_x", "Real_ID_y", "Mode", "Text"]]

    pre_bow.to_csv('Bag_of_Words_df_Single_Sen.tsv', sep='\t', index=False)

    print "\n", "Done with Bag of Words data frame", "\n"

    return pre_bow

def manual_trte_split(bow_df, random_state, test_set_prop):
    """Split BOW Data Frame so no two interactions occur both in Test and in Train sets
    The code will try to get the around the proportion given by test_set_prop where train set
    will be 1-test_set_prop."""

    np.random.seed(random_state)

    #Get a Full list of proteins
    real_x = bow_df["Real_ID_x"].tolist()
    real_y = bow_df["Real_ID_y"].tolist()
    full_protein_list = list(set(real_x + real_y)) #A set because it is faster to search it!

    train_size = len(full_protein_list) - int(len(full_protein_list)*test_set_prop)
    train_prots = set(np.random.choice(np.array(full_protein_list), train_size, replace = False))
    test_prots = set(full_protein_list) - set(train_prots)

    col_names = ["Real_ID_x", "Real_ID_y", "Mode", "Text"]
    train = pd.DataFrame(columns=col_names)
    test = pd.DataFrame(columns=col_names)

    for _, row in bow_df.iterrows():
        if row["Real_ID_x"] in train_prots and row["Real_ID_y"] in train_prots:
            temp_df = pd.DataFrame(row).T #Dirty but works
            train = train.append(temp_df)
            del temp_df
        elif row["Real_ID_x"] in test_prots and row["Real_ID_y"] in test_prots:
            temp_df = pd.DataFrame(row).T #Dirty but works
            test = test.append(temp_df)
            del temp_df
        else:
            pass

    return train.reset_index(drop=True), test.reset_index(drop=True)

def texts_to_words(raw_text):
    """Function to convert a raw text to a string of words
       The input is a single string (a raw text), and
       the output is a single string (a preprocessed text)"""

    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 4. Remove stop words
    meaningful_words = [w for w in words if w not in stops]
    #
    # 5. Join the words back into one string separated by space,
    # and return the result.
    return(" ".join(meaningful_words))

def bag_of_words_feat_vecs(data_train, data_test, n_features):
    """Adding later"""

    # Get the number of reviews based on the dataframe column size
    num_texts = data_train["Text"].size

    # Initialize an empty list to hold the clean reviews
    print "Cleaning and parsing the training set article sentences...\n"
    clean_train_texts = []
    for i in xrange(0, num_texts):
        clean_train_texts.append(texts_to_words(data_train["Text"][i]))

    print "Creating the bag of words...\n"
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=n_features)
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
    for i in xrange(0, num_texts):
        # if( (i+1) % 1000 == 0 ):
            # print "Review %d of %d\n" % (i+1, num_texts)
        clean_texts = texts_to_words(data_test["Text"][i])
        clean_test_texts.append(clean_texts)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_texts)
    test_data_features = test_data_features.toarray()

    return train_data_features, test_data_features

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

def RF_classifier(train_vector, test_vector, labels_train, labels_test, n_trees, ran_s):
    """Adding Later"""
    print "Training the random forest..."

    # Initialize a Random Forest classifier with n_trees trees
    forest = RandomForestClassifier(n_estimators=n_trees, random_state=ran_s)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit(train_vector, labels_train)
    probs = forest.predict_proba(test_vector)[:, 1] #For ROC Curve

    # Use the random forest to make sentiment label predictions
    print "Predicting based on model..."
    result = forest.predict(test_vector)

    error = get_accuracy(result, labels_test)

    return error, probs


