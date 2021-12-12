#this file solves through 1f
import pandas
import numpy

from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as pyplot
#I use KMedoids for cluster analysis, so I import that
from sklearn_extra.cluster import KMedoids
#I want to compare it to KMeans, so I import that as well
from sklearn.cluster import KMeans
#import silhouette_score to find how good the clusters are too
from sklearn.metrics import silhouette_score

#importing our csv
dataset  = pandas.read_csv("dataset_final.csv")

dataset.drop(['country'], axis=1, inplace=True)
#dropping the index column
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
#dropping missing data
dataset.dropna(inplace=True)

print(dataset)
#these next chunks let me find how many factors to use, solving 1c
machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
pyplot.scatter(range(1,dataset.shape[1]+1), ev)
pyplot.savefig("plot.png")
pyplot.close()
#it looks like the first 4 factors matter the most so I analyze those 4

machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(dataset)
output = machine.loadings_
numpy.set_printoptions(suppress=True)
print(output)
#nall 4 seem significant so I'll stick with it
#1c: the answer is 4 factors

#now I make a factor loadings matrix with 4 factors
machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(dataset)
#I save my matrix to loadings
loadings = machine.loadings_
#and print my matrix
print("\nfactor loadings:\n")
print(loadings)
print(machine.get_factor_variance())

#this line gives me my dataset for clustering
dataset = dataset.values
#and this line puts that in a matrix form with my factors
result = numpy.dot(dataset, loadings)
#print results to make sure my obervations are correct
print(result)
print(result.shape)
#it shows I have 21640 observations and 4 factors, yay

#1d: I begin solving 1d with the following code
#I define run_kmedoids(n) to make scatterplots with n clusters
def run_kmedoids(n):
	machine = KMedoids(n_clusters=n)
#we use result since it is already in matrix format
	machine.fit(result)
	results = machine.predict(result)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	silhouette = 0
	if n>1:
		silhouette = silhouette_score(result, machine.labels_, metric = 'euclidean')
	pyplot.scatter(result[:,0], result[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("KMedoidsplot_color_" + str(n) + ".png")
	pyplot.close()
	#these lines give me my Silhouette Score and SSD to measure how good my KMedoids clusters are
	print(n)
	print("KMedoids Silhouette Score = %f" %silhouette)
	print("KMedoids SSD = %f" %ssd)
#the following lines let me compare the results of different clusters, and find the best number
run_kmedoids(1)
run_kmedoids(2)
run_kmedoids(3)
#this one seems to be the best, so I the SSD and Silhouette score for comparison
#1e: 3 clusters is optimal in this case
#I find Silhouette Score is .488748
#I find SSD is 156850.75
run_kmedoids(4)
run_kmedoids(5)

#1d: this chunk of code completes 1d. I used KMedoids and KMeans
#repeating that same process for KMeans
def run_kmeans(n):
	machine = KMeans(n_clusters=n)
#we use result since it is already in matrix format
	machine.fit(result)
	results = machine.predict(result)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	silhouette = 0
	if n>1:
		silhouette = silhouette_score(result, machine.labels_, metric = 'euclidean')
	pyplot.scatter(result[:,0], result[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("KMeansplot_color_" + str(n) + ".png")
	pyplot.close()
	#these lines give me my Silhouette Score and SSD to measure how good my KMeans clusters are
	print(n)
	print("KMeans Silhouette Score = %f" %silhouette)
	print("KMeans SSD = %f" %ssd)

#I try different numbers of clusters to find the best one again
run_kmeans(1)
run_kmeans(2)
run_kmeans(3)
#once again, 3 is the best number of clusters, so I record Silhouette Score and SSD
#1e: 3 clusters is optimal in this case
#I find Silhouette Score is .49045
#I find SSD is 1694082.91
run_kmeans(4)
run_kmeans(5)


#1f: It is unclear...
#I can compare my 3-cluster results directly for KMedoids and KMeans
#I find that my cluster with KMeans has a better value than KMedoids for Silhouette Score, but a worse value for SSD
#therefore, it is not clear which algorithm works better
