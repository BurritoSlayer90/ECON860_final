#this file lets me answer 1g, as I compare my results from a country to the whole dataset
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

#to solve 1g, I repeat the previous process for different countries
#here i do this for GB

#creating a new dataset with entries from GB only
datasetgb  = pandas.read_csv("dataset_final.csv")
#empty df to store our rows from GB
dfgb = pandas.DataFrame()
#this for loop appends every row that contains an entry from GB
for rows in range(len(datasetgb)):
	if(datasetgb.country[rows] == 'GB'):
		dfgb = dfgb.append(datasetgb.loc[rows],ignore_index = True)

#now we repeat the same procedure as in final.py to turn it into a usable matrix
dfgb.drop(['country'], axis=1, inplace=True)
#dropping the index column
dfgb.drop(['Unnamed: 0'], axis=1, inplace=True)
#dropping missing data
dfgb.dropna(inplace=True)

print(dfgb)
#I repeat the same procedure to see how many factors to use...
machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(dfgb)
ev, v = machine.get_eigenvalues()
print(ev)

pyplot.scatter(range(1,dfgb.shape[1]+1), ev)
pyplot.savefig("GBplot.png")
#the data looks similar to the full set, but more sparse
pyplot.close()
#it looks like the first 4 factors matter the most so I repeat with 7 factors
machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(dfgb)
output = machine.loadings_
numpy.set_printoptions(suppress=True)
print(output)
#all 4 factors seem significant here, so we'll stick with 4 factors
#1c: the answer is 4 factors

#now I make a factor loadings matrix with 4 factors
machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(dfgb)
#I save my matrix to loadings
loadings = machine.loadings_
#and print my matrix
print("\nfactor loadings:\n")
print(loadings)
print(machine.get_factor_variance())

#this line gives me my dataset for clustering
dfgb = dfgb.values
#and this line puts that in a matrix form with my factors
result = numpy.dot(dfgb, loadings)
#print results to make sure my obervations are correct
print(result)
print(result.shape)
#it shows I have 1613 observations and 4 factors, yay

#creating a basic scatterplot for my data
pyplot.scatter(result[:,0], result[:,1])
pyplot.savefig("scatterplot.png")

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
	pyplot.savefig("GB KMedoidsplot_color_" + str(n) + ".png")
	pyplot.close()
	#these lines give me my Silhouette Score and SSD to measure how good my KMedoids clusters are
	print(n)
	print("GB KMedoids Silhouette Score = %f" %silhouette)
	print("GB KMedoids SSD = %f" %ssd)
#the following lines let me compare the results of different clusters, and find the best number
run_kmedoids(1)
run_kmedoids(2)
run_kmedoids(3)
#this one seems to be the best, so I the SSD and Silhouette score for comparison
#1e: 3 clusters is optimal in this case
#I find Silhouette Score is .494847
#I find SSD is 11622.616724
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
	pyplot.savefig("GB KMeansplot_color_" + str(n) + ".png")
	pyplot.close()
	#these lines give me my Silhouette Score and SSD to measure how good my KMeans clusters are
	print(n)
	print("GB KMeans Silhouette Score = %f" %silhouette)
	print("GB KMeans SSD = %f" %ssd)

#I try different numbers of clusters to find the best one again
run_kmeans(1)
run_kmeans(2)
run_kmeans(3)
#once again, 3 is the best number of clusters, so I record Silhouette Score and SSD
#1e: 3 clusters is optimal in this case
#I find Silhouette Score is .494796
#I find SSD is 120725.14
run_kmeans(4)
run_kmeans(5)


#1f: KMedoids!
#I can compare my 3-cluster results directly for KMedoids and KMeans
#I find that my cluster with KMedoids has a better Silhouette Score and SST than KMeans 

#1g: Yes. We find that GB differs from the collective dataset, so other countries must vary too
