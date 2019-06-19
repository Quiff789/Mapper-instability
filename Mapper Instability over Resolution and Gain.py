
import kmapper as km
from random import shuffle
import re
import numpy as np
import itertools
import math
import sys
import os
import copy
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

Original_Set = sys.stdout

# Disable Print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore Print
def enablePrint():
    sys.stdout = Original_Set#sys.__stdout__

# Initialize Kepler Mapper
mapper = km.KeplerMapper(verbose=1)

#**************************functions**************************

#Puts cluster keys in to number form,
#records clusters in each cube and how many points in each clsuter
def Relabel(Nodes,Sequence_Number,Shift,Bins,Points_Size):
    Clusters = dict()
    Cluster_Numbers = [0] * Bins
    All = set(range(Points_Size))
    for Key in Nodes.keys():
        #Finds numbers in text for the Bin and cluster in the Bin
        temp = re.findall(r'\d+', Key)
        Key_Cluster = (int(temp[0]),int(temp[1]))
        #Changes the maximla cluster number if necessary
        if Cluster_Numbers[Key_Cluster[0]] < Key_Cluster[1]+1:
            Cluster_Numbers[Key_Cluster[0]] = Key_Cluster[1]+1
        #Gives the points the correct lable after subsample shift and place them in as set
        for Point in Nodes[Key]:
            #Labels the point with the correct number given the shifting
            if Point >= Sequence_Number*Shift:
                Real_Point=Point+Shift
            else:
                Real_Point=Point
            #Removes the labels of the relabled point numbers from the set of all points not yet removed
            if Key_Cluster in Clusters:
                Clusters[Key_Cluster].add(Real_Point)
            else:
                Clusters[Key_Cluster]=set([Real_Point])
    #Lists the size of each cluster in each clube
    Cluster_Sizes = []
    for i in range(Bins):
        Cluster_Sizes.append([])
    for Cluster in Clusters:
        Cluster_Sizes[Cluster[0]].append((len(Clusters[Cluster]),Cluster))
    return(Clusters,Cluster_Numbers,Cluster_Sizes)

#Backtracking recurtion for computing upper bound only on clustering within cubes
def Bound_Perm(A,B,Cube,Unused,Perm,Value,Current_Mismatch,Cube_Mismatch):
    Unused_Local = copy.deepcopy(Unused)
    Mismatch_Local = copy.deepcopy(Current_Mismatch)
    Num = Perm - len(Unused_Local)
    for i in Unused_Local:
        Unused_Next = copy.deepcopy(Unused_Local)
        Unused_Next.remove(i)
        if (Cube,i) in B[0]:
            Mismatch_Next = Mismatch_Local.union(A[0][(Cube,Num)] ^ B[0][(Cube,i)])
        else:
            Mismatch_Next = Mismatch_Local.union(A[0][(Cube,Num)])
        Total = len(Mismatch_Next)
        if Total < Value:
            if Unused_Next == []:
                Value = Total
                Cube_Mismatch = Mismatch_Next
            else:
                Value, Cube_Mismatch = Bound_Perm(A,B,Cube,Unused_Next,Perm,Value,Mismatch_Next,Cube_Mismatch)                        
    return Value, Cube_Mismatch 

#Find the clustering insability values then uses thier choics to obtain an upper bound on the Mapper instability
def Upper_Bound(Output_Pair,Bin_Number,Max_Clusters,Missing,Points):

    Bound_Mismatch = set()

    #Clustering choices
    for Cube in range(Bin_Number):                             
        Perms = list(range(Output_Pair[Max_Clusters[Cube][0]][1][Cube]))
        A = Output_Pair[Max_Clusters[Cube][0]]
        B = Output_Pair[Max_Clusters[Cube][1]]
        Value, Cube_Mismatch = Bound_Perm(A,B,Cube,Perms,len(Perms),Points,Missing,set())
        #Compute Mapper mismatch for best clustering mismatches
        Bound_Mismatch = Bound_Mismatch.union(Cube_Mismatch)

    Bound = len(Bound_Mismatch)
    return Bound


#Minimal matching distence for two Mapper Outputs (on points that coincide)
def Matching_Distence(Out1,Out2,Bin_Number,Points,Missing):
    
    Output_Pair=[Out1,Out2]
    Max_Clusters = [0] * Bin_Number
    Max_Perm_Number = [0] * Bin_Number
    Cluster_set = []
    
    #Within each Bin finds the Mapper output with the most clusters, 
    #recorsd this and the clusters within it
    for i in range(Bin_Number):
        if Out1[1][i] > Out2[1][i]:
            Max_Clusters[i] = [0,1]
        else:
            Max_Clusters[i] = [1,0]
        Max_Perm_Number[i] = Output_Pair[Max_Clusters[i][0]][1][i]
        Cluster_set = Cluster_set + Output_Pair[Max_Clusters[i][0]][2][i]

    Bound = Upper_Bound(Output_Pair,Bin_Number,Max_Clusters,Missing,Points)
    
    #Sorts the clusters (largest first)
    Sorted_Clusters = list(reversed(sorted(Cluster_set, key=lambda Cluster: Cluster[0])))

    Value = Bound
    Not_Finished = True
    Backtrack = False
    Cluster = 0
    Matching = 0
    Final_Cluster = len(Sorted_Clusters)
    Mismatch = Missing
    Choices_Matching = []
    Choices_Mismatch =[]
    Choices_Cubes = {}
    for i in range(Bin_Number):
        Choices_Cubes[i] = set()
    
    #Computes the distnece for the present permutation
    while Not_Finished:
        Backtrack = False
        #Trys each cluster intersection for the current Cluster Matching
        #If it is not a valid permutation or the mismatch goes above current best then backtrack
        Cube_Current = Sorted_Clusters[Cluster][1][0]
        Cluster_Current = Sorted_Clusters[Cluster][1][1]

        #Check if the curent choice is valid if not move to the next choice
        #otherwise enters the choice and adds the mismatch
        if Matching in Choices_Cubes[Cube_Current]:
            Matching += 1
            Backtrack = True
        else:
            #Compute mismatch
            Source = Output_Pair[Max_Clusters[Cube_Current][0]]
            Target = Output_Pair[Max_Clusters[Cube_Current][1]]
            if (Cube_Current,Matching) in Target[0]:
                Current_Mismatch = Source[0][(Cube_Current,Cluster_Current)] ^ Target[0][(Cube_Current,Matching)]
            else:
                Current_Mismatch = Source[0][(Cube_Current,Cluster_Current)]
            New_Mismatch = Mismatch.union(Current_Mismatch) 
            Total = len(New_Mismatch)
            if Total >= Value:
                Matching += 1
                Backtrack = True
            #Otherwise next Cluster, if final cluster record value and increment matching
            else:
                if Cluster == Final_Cluster-1:
                    Value = Total
                    Backtrack = True
                else:
                    Cluster += 1
                    Choices_Mismatch.append(Mismatch)
                    Mismatch = New_Mismatch
                    Choices_Matching.append(Matching)
                    Choices_Cubes[Cube_Current].add(Matching)
                    Matching = 0
            
        #If necessary bactack and check for termination
        while Backtrack:
            if Matching >= Max_Perm_Number[Cube_Current]:
                if Cluster == 0:
                    Not_Finished = False
                    Backtrack = False
                else:
                    Cluster -= 1
                    Cube_Current = Sorted_Clusters[Cluster][1][0]
                    Mismatch = Choices_Mismatch[-1]
                    Choices_Cubes[Cube_Current].discard(Choices_Matching[-1])
                    Matching = Choices_Matching[-1] + 1
                    del Choices_Matching[-1]
                    del Choices_Mismatch[-1]
            else:
                Backtrack = False
    #Retunes the minimal distence amoung all permutations of the clusters and the upper bound
    return [Value,Bound] 


#Instability of a list of Mapper networks [...[Netork,Cluster numbers]...]
def Instability(Outputs,Bin_Number,Shift,Points):
    l=len(Outputs)
    Values = []
    Bounds = []
    for i in range(l):
        for j in range(i+1,l):
            Missing = set(list(range(i*Shift,(i+1)*Shift)) + list(range(j*Shift,(j+1)*Shift)))
            [Value, Bound] = Matching_Distence(Outputs[i],Outputs[j],Bin_Number,Points,Missing)
            Values.append(Value-2*Shift)
            Bounds.append(Bound-2*Shift)
    Instab = (2*sum(Values))/(l*(l-1)*(Points-2*Shift))
    Bound = (2*sum(Bounds))/(l*(l-1)*(Points-2*Shift))
    return [Instab,Bound]

#**************************Main Program**************************

#Percentage Overlap increments
temp1=list(range(25,501,25))
temp2=[]
for temp in temp1:
    temp2.append(temp/1000)

#Full settings
Cubes_List = list(range(2,22,1))
Overlap_List = temp2
Shift = 100
Points = 10*Shift
Runs = 10

Epsilon = 0.15

Noise = 0.055

#Printing disabled
blockPrint()

#Open text file
Text_file = open("Stabiltiy Results, Cubes:"+str((min(Cubes_List),max(Cubes_List),Cubes_List[1]-Cubes_List[0]))+", Overlap:"+str((min(Overlap_List),max(Overlap_List),Overlap_List[1]-Overlap_List[0]))+", Noise: "+str(Noise)+", Epsilon"+ str(Epsilon) + ", Shift and Points" + str((Shift,Points)) + ", Runs" + str(Runs) +".txt","w+")

#Random data points form two consentric circles
data_set, labels = datasets.make_circles(n_samples=Points, noise=Noise, factor=0.3)
Plot = [[],[]]
for i in range(len(data_set)):
    Plot[0].append(data_set[i][0])
    Plot[1].append(data_set[i][1])

Line = 0

Instabilities = []

#Mapper and stability analysis
for k in range(len(Cubes_List)):

    Instabilities.append([])
    
    for l in range(len(Overlap_List)):

        Values = []
        Line += 1
            
        for r in range(Runs):

            Cubes = Cubes_List[k]
            Overlap = Overlap_List[l]

            #Printing disabled
            blockPrint()

            #Randomize data
            A=list(range(len(data_set)))
            shuffle(A)
            data=[]
            for a in A:
                data.append(data_set[a])
            data = np.array(data_set)
                
            lens = []
            for d in data:
                lens.append([(d[1]+1.5)/3])
            lens = np.array(lens)

            if r == 0 and False:
                #Plots data with bouandries and overlap
                plt.clf()
                plt.plot(Plot[0], Plot[1], 'bx')
                plt.axis('equal')
                for i in range(Cubes-1):
                    Hight = -1.5+(i+1)*3/Cubes
                    Overlap_Shift = Overlap*3/Cubes
                    plt.plot([-1.5,1.5],[Hight,Hight],'r-')
                    plt.plot([-1.5,1.5],[Hight+Overlap_Shift,Hight+Overlap_Shift],'r--')
                    plt.plot([-1.5,1.5],[Hight-Overlap_Shift,Hight-Overlap_Shift],'r--')
                plt.savefig("Data Set with Bins, Cubes"+str(Cubes)+", Overalp"+str(Overlap)+".png")
                
            Samples = []
            Lenses = []
            
            #generates samples and corresponding lenses
            for i in range(int(Points/Shift)):
                Samples.append(np.array(list(data[0:i*Shift])+list(data[(i+1)*Shift:Points])))
                Lenses.append(np.array(list(lens[0:i*Shift])+list(lens[(i+1)*Shift:Points])))

            #Mapper for whole complex
            Full_Complex = mapper.map(lens, X=data,
                clusterer=sklearn.cluster.DBSCAN(eps=Epsilon, min_samples=0),
                cover=km.Cover(n_cubes=Cubes, perc_overlap=Overlap))

            if r == 0:
                #Creats visual file
                mapper.visualize(Full_Complex, path_html="Full Complex, Cubes " + str(Cubes) + ", Overlap " + str(Overlap) + ", Epsilon" + str(Epsilon) + ", Points " + str(Points) + ", Noise " + str(Noise) + ".html",
                    custom_meta={'Data': "datasets.make_circles(n_samples=" + str(Points)},
                    custom_tooltips=labels)
                
            Mapper_Outputs=[]
                
            #Mapper for samples, lables samples and recores maximum cluster numbers
            for s in range(len(Samples)):
                Simplicial_Complex = mapper.map(Lenses[s], X=Samples[s],
                    clusterer=sklearn.cluster.DBSCAN(eps=Epsilon, min_samples=0),
                    cover=km.Cover(n_cubes=Cubes, perc_overlap=Overlap))
                
                Clusters, Cluster_Numbers, Cluster_Sizes = Relabel(Simplicial_Complex["nodes"],s,Shift,Cubes,Points)
                Mapper_Outputs.append([Clusters,Cluster_Numbers,Cluster_Sizes])
                if r == 0 and False:
                #Creats visual file
                    mapper.visualize(Simplicial_Complex, path_html="Sample, Cubes " + str(Cubes) + ", Overlap " + str(Overlap)+ ", Epsilon" + str(Epsilon) + ", Noise " + str(Noise) + ", Sample " + str(s) + ".html",
                        custom_meta={'Sample '+ str(s): "datasets.make_circles(n_samples=" + str(Points-Shift)},
                        custom_tooltips=labels)
                        
            [Value,Bound] = Instability(Mapper_Outputs,Cubes,Shift,Points)
            Values.append(Value)

        #Printing enabled
        enablePrint()

        Value = sum(Values)/len(Values)

        #Prints the current instability value and writes them to a text file
        print("*************************")
        String = "Cubes: " + str(Cubes) + ", Overlap: " + str(Overlap) + ", Epsilon:" + str(Epsilon) + ", Average Instability: " + str(Value)
        print(String)
        Text_file.write(String + "%d\r\n" % (Line))

        #Record instability for surface plot
        Instabilities[k].append(Value)
                
#Plots data
plt.clf()
plt.plot(Plot[0], Plot[1], 'bx')
plt.axis('equal')
plt.savefig("Data Set.png")


#Contor plot of instability surface
Overlap_Persentage = []
for Over in Overlap_List:
    Overlap_Persentage.append(Over*100)

y = np.array(Cubes_List)
x = np.array(Overlap_Persentage)
X, Y = np.meshgrid(x, y)
Z = np.array(Instabilities)

plt.figure()
cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)

plt.title('Instability surface resolution vs gain')
plt.ylabel('Number of bins')
plt.xlabel('Percentage overlap')

plt.savefig("Averages Plot of Instabilitys, Cubes:" + str((min(Cubes_List),max(Cubes_List),Cubes_List[1]-Cubes_List[0])) + ", Overlap:"+str((min(Overlap_List),max(Overlap_List),Overlap_List[1]-Overlap_List[0])) + ", Epsilon " + str(Epsilon) +", Noise " + str(Noise) + ".png")

#Find local minima
Local_Minima = []
for i in range(len(Cubes_List)):
    for j in range(len(Overlap_Persentage)):
        Minima = True
        if i != len(Cubes_List) - 1:
            if Instabilities[i][j] > Instabilities[i+1][j]:
                Minima = False
        if i != 0:
            if Instabilities[i][j] > Instabilities[i-1][j]:
                Minima = False
        if j != len(Overlap_Persentage) - 1:
            if Instabilities[i][j] > Instabilities[i][j+1]:
                Minima = False
        if j != 0:
            if Instabilities[i][j] > Instabilities[i][j-1]:
                Minima = False
        if Minima == True:
            Local_Minima.append((Overlap_Persentage[j],Cubes_List[i]))

#Plot local minima
for point in Local_Minima:
    plt.plot(point[0], point[1], 'rx')

plt.savefig("Averages Plot of Instabilitys with Local Minima points" + ".png")

Line += 1

print("Local Minima at: " + str(Local_Minima))
Text_file.write("Local Minima at: " + str(Local_Minima) + "%d\r\n" % (Line))

#Closes teh text file
Text_file.close()
