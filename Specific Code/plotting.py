from __future__ import division
import pandas as pd
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib import style

#### TO RUN ####
# python plotting.py #
# IN A FOLDER WITH Full_Models.tsv#
###############

style.use('fivethirtyeight')

def open_model(model):
    
    model_df = pd.read_csv(model, sep = "\t", names = ["Classification Model", "Feature Count", "Error", "Time"])
    
    return model_df

def main():

	model_df = open_model("./Results/Full_Models.tsv")

	fig, ax = plt.subplots()
	ax.xaxis.set_ticks(np.arange(0, 12000, 1000))
	axes = plt.gca()
	axes.set_xlim([0,11000])

	for i,r in model_df.iterrows():
		if r["Classification Model"] == "Random Forest (100 trees, random state = 23)":
			ran_for = ax.scatter(r["Feature Count"], r["Error"], s=200, marker = 'o', c = 'b', label = r["Classification Model"])
			ax.annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
		if r["Classification Model"] == "Neural Network":
			nn = ax.scatter(r["Feature Count"], r["Error"], s=200, marker = '^', c = 'r', label = r["Classification Model"])
			ax.annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
		if r["Classification Model"] == "Logistic Regression":
			log_reg = ax.scatter(r["Feature Count"], r["Error"], s=200, marker = 'v', c = 'g', label = r["Classification Model"])
			ax.annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
		if r["Classification Model"] == "Support Vector Machines":
			svm = ax.scatter(r["Feature Count"], r["Error"], s=200, marker = 's', c = 'purple', label = r["Classification Model"])
			ax.annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
		if r["Classification Model"] == "K Nearest Neighbor (5)":
			knn = ax.scatter(r["Feature Count"], r["Error"], s=200, marker = '>', c = 'orange', label = r["Classification Model"])
			ax.annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))

	plt.title("Classification Model Comparison")
	plt.xlabel('Bag of Words Feature Count')
	plt.ylabel('Error')
	# plt.colorbar()
	plt.legend(handles = [ran_for, nn, log_reg, svm, knn], loc = 4, prop={'size':15})
	plt.show()

	# fig, ax = plt.subplots(2,3)

	# for i,r in model_df.iterrows():
	# 	if r["Classification Model"] == "Random Forest (100 trees, random state = 23)":
	# 		ran_for = ax[0,0].scatter(r["Feature Count"], r["Error"], s=200, marker = 'o', c = 'b', label = r["Classification Model"])
	# 		ax[0,0].annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
	# 		ax[0,0].set_ylim([0.15,0.17])
	# 		ax[0,0].set_ylabel("Error (0.15 - 0.17)")
	# 		ax[0,0].set_title("Random Forests")
	# 	if r["Classification Model"] == "Neural Network":
	# 		nn = ax[0,1].scatter(r["Feature Count"], r["Error"], s=200, marker = '^', c = 'r', label = r["Classification Model"])
	# 		ax[0,1].annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
	# 		ax[0,1].set_ylim([0.17,0.19])
	# 		ax[0,1].set_ylabel("Error (0.17 - 0.19)")
	# 		ax[0,1].set_title("Neural Network")
	# 	if r["Classification Model"] == "Logistic Regression":
	# 		log_reg = ax[0,2].scatter(r["Feature Count"], r["Error"], s=200, marker = 'v', c = 'g', label = r["Classification Model"])
	# 		ax[0,2].annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
	# 		ax[0,2].set_ylim([0.17,0.19])
	# 		ax[0,2].set_ylabel("Error (0.17 - 0.19)")
	# 		ax[0,2].set_title("Logistic Regression")
	# 	if r["Classification Model"] == "Support Vector Machines":
	# 		svm = ax[1,0].scatter(r["Feature Count"], r["Error"], s=200, marker = 's', c = 'purple', label = r["Classification Model"])
	# 		ax[1,0].annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]), textcoords = "figure points")
	# 		ax[1,0].set_ylim([0.19,0.21])
	# 		ax[1,0].set_ylabel("Error (0.20 - 0.21)")
	# 		ax[1,0].set_title("Support Vector Machines")
	# 	if r["Classification Model"] == "K Nearest Neighbor (5)":
	# 		knn = ax[1,1].scatter(r["Feature Count"], r["Error"], s=200, marker = '>', c = 'orange', label = r["Classification Model"])
	# 		ax[1,1].annotate("%.3f" % r["Time"], (r["Feature Count"], r["Error"]))
	# 		ax[1,1].set_title("K Nearest Neighbor")
	# plt.show()


if __name__ == "__main__":
	main()