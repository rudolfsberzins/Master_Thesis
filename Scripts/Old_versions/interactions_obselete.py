import pandas as pd
from sys import argv
import re
import numpy as np
# import nltk.data
# import glob
# import os


def open_tsv(tsv_name):
	"""Opens a TSV File Containing the Interactions from STRING DB

	INPUT: Name of the tsv file

	OUTPUT: Pandas DataFrame of the .tsv file"""
	
	tsv_file = pd.read_csv(tsv_name, sep = '\t')

	return tsv_file

def filter_interactions(int_df):
	"""Filters the interactions file to only contain certain interactions, e.g. binding

	INPUT: Interactions Data Frame

	OUTPUT: Filtered Data Frame containing only bindings"""

	desired_interac = raw_input("What interaction would you like to get?: ")

	int_df["item_id_a"] = int_df["item_id_a"].str[5:]
	int_df["item_id_b"] = int_df["item_id_b"].str[5:]

	interaction = int_df[int_df["mode"] == str(desired_interac)]
	if interaction.empty:
		print "\n"
		print "There are no interactions of that type."
		print "\n"
	else:
		not_interaction = int_df[int_df["mode"] != str(desired_interac)]

	
	return interaction, not_interaction, desired_interac

def main():

	_, inta = argv

	interactions = open_tsv(inta)
	filtered, filtered_oposite, d_i = filter_interactions(interactions)

	filtered.to_csv('Filtered_Interaction_'+d_i+'.tsv', sep = '\t', index = False)
	filtered_oposite.to_csv('Oposite_Filtered_Interaction_'+d_i+'.tsv', sep = '\t', index = False)


if __name__ == "__main__":
	main()