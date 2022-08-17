"""This script computes a single Mapper Instability value on a concentric circles data."""

import kmapper as km
from random import shuffle
import numpy as np
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt

import MapperInstability as I

#Mapper Settings
Cubes = 10
Overlap = 0.2
#Number of data points
Points = 10000
#number of datapoints per-sample (should divide the number of points)
Shift = 100
#clustering epsilon
Epsilon = 0.15

#Initialize Kepler Mapper
mapper = km.KeplerMapper(verbose = 0)

#Samples random data points form two consentric circles
data, labels = datasets.make_circles(n_samples = 1000, noise = 0.055, factor = 0.3)

#generates samples and corresponding lenses for Keppler Mapper    
Samples = []
Lenses = []
lens = (data[:,1] + 1.5) / 3
for i in range(int(Points/Shift)):
    Samples.append(np.array(list(data[0:i*Shift])+list(data[(i+1)*Shift:Points])))
    Lenses.append(np.array(list(lens[0:i*Shift])+list(lens[(i+1)*Shift:Points])))

#Mapper for whole complex
Full_Simplicial_Complex = mapper.map(lens, X = data,
                          clusterer=sklearn.cluster.DBSCAN(eps=Epsilon, min_samples=0),
                          cover=km.Cover(n_cubes = Cubes, perc_overlap = Overlap))


#Creates visual file of the Mapper graph
mapper.visualize(Full_Simplicial_Complex, path_html = "Mapper graph.html")
               
#Mapper for samples complexes individualy
Mapper_Outputs = []
for s in range(len(Samples)):
    #get Mapper graph for each subsample
    Simplicial_Complex = mapper.map(Lenses[s], X=Samples[s],
                                    clusterer = sklearn.cluster.DBSCAN(eps=Epsilon, min_samples = 0),
                                    cover = km.Cover(n_cubes = Cubes, perc_overlap = Overlap))
    #generate clustr infomation for each sample            
    Clusters, Cluster_Numbers, Cluster_Sizes = I.Relabel(Simplicial_Complex["nodes"], s, Shift, Cubes, Points)
    Mapper_Outputs.append([Clusters, Cluster_Numbers, Cluster_Sizes])

#Mapper and instability computation                       
Instability = I.Instability(Mapper_Outputs, Cubes, Shift, Points)
print("\n Instability: " + str(Instability))


                





