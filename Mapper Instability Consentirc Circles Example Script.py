"""This script performs the Mapper Instability experiment on the concentric circles data set present in
the paper: Francisco Belch'{\i}, Jacek Brodzki, Matthew Burfitt, Mahesan Niranjan,
"A numerical measure of the instability of Mapper-type algorithms" Here: https://arxiv.org/abs/1906.01507v1"""

import kmapper as km
from random import shuffle
import numpy as np
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt

import MapperInstability as I

#Percentage Overlap increments
temp1=list(range(25,501,25))
temp2=[]
for temp in temp1:
    temp2.append(temp/1000)

#Full settings
Cubes_List = [2,4]#list(range(2,22,1))
Overlap_List = [0.2,0.4]#temp2
Shift = 100
Points = 10*Shift
Runs = 1#10
Epsilon = 0.15
Noise = 0.055

#Initialize Kepler Mapper
mapper = km.KeplerMapper(verbose = 0)

#Open text file to save results
Text_file = open("Stabiltiy Results", "w+")

#Samples Random data points form two consentric circles
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

            #Randomize data
            A=list(range(len(data_set)))
            shuffle(A)
            data=[]
            for a in A:
                data.append(data_set[a])
            data = np.array(data_set)
                
            lens = (data[:,1] + 1.5) / 3

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
                clusterer = sklearn.cluster.DBSCAN(eps=Epsilon, min_samples=0),
                cover = km.Cover(n_cubes=Cubes, perc_overlap=Overlap))

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
                
                Clusters, Cluster_Numbers, Cluster_Sizes = I.Relabel(Simplicial_Complex["nodes"],s,Shift,Cubes,Points)
                Mapper_Outputs.append([Clusters, Cluster_Numbers, Cluster_Sizes])
                if r == 0 and False:
                #Creates visual file
                    mapper.visualize(Simplicial_Complex, path_html="Sample, Cubes " + str(Cubes) + ", Overlap " + str(Overlap)+ ", Epsilon" + str(Epsilon) + ", Noise " + str(Noise) + ", Sample " + str(s) + ".html",
                        custom_meta={'Sample '+ str(s): "datasets.make_circles(n_samples=" + str(Points-Shift)},
                        custom_tooltips=labels)
                        
            Value = I.Instability(Mapper_Outputs, Cubes, Shift, Points)
            Values.append(Value)

        Value = sum(Values)/len(Values)

        #Prints the current instability value and writes them to a text file
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

plt.savefig("Averages Plot of Instabilitys.png")

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
