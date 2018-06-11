import pandas as pd
import numpy as np
from numpy import genfromtxt
import time
import copy
import sys

class Node(object):
    def __init__(self, label):
        self.label = label
        self.left = None
        self.right = None

    #Child must be of type Node
    def getLeft(self):
        return self.left
    def getRight(self):
        return self.right
    def getLabel(self):
        return self.label

def agglomerative(D):


    #og = np.arange(25000000).reshape(5000,5000)
    df = np.triu(D, 1)

    C = []
    currentclusters = []

    ClusterCount = 5000
    Ci = []
    for x in range(df.shape[0]):
        Ci.append([x])
    MinCluster = []
    C.append(Ci)
    #print(Ci)
    hahah = 5000
    while df.shape[0] > 2:
        #start = time.time()
        i,j = np.where(df==np.min(df[np.nonzero(df)]))
        i = i[0]
        j = j[0]
        hahah -= 1
        mindist = df[i,j]
        #print(df[0,0])
        #print(mindist)
        hello = list(range(0,df.shape[0]))
        #print(hello)
        hello.remove(i)
        hello.remove(j)

        Cnext = [Ci[x] for x in hello]
        Cnext.append(Ci[int(i)] +  Ci[int(j)])
        MinCluster.append((Ci[int(i)] +  Ci[int(j)], float(mindist)))
        Ci = Cnext
        C.append(Ci)

        ello = [[x] for x in hello]
        newmatrix = np.array(df[ello, hello])
        b = np.array(np.zeros(newmatrix.shape[0]))
        yo = np.vstack((newmatrix,b))

        newclustercol = []

        for x in hello:
            potentialval = (df[i,x], df[x,i], df[j,x], df[x,j])
            potentialval = list(filter(lambda a: a != 0, potentialval))
            newclustercol.append(float(max(potentialval)))

        newclustercol.append(0.0)
        sos = np.asarray(newclustercol).reshape(yo.shape[0],1)
        yeet = np.concatenate([yo,sos], axis=1)
        df = yeet

    MinCluster.append((Ci[int(0)] +  Ci[int(1)], float(np.max(df))))
    C.append(Ci[int(0)] +  Ci[int(1)])


    return C, MinCluster

def buildTree(index, D):
    height = D[index][1]
    N = Node(height)
    Z = D[index][0]

    if len(Z) == 2:
        N.left = Node(Z[0])
        N.right = Node(Z[1])
        return N

    for i in range(index-1, -1, -1):
        Y = D[i][0]
        flag = True
        for y in Y:
            if y not in Z:
                flag = False
        if flag == True:
            yindex = i
            break

    X = [z for z in Z if z not in Y]
    if len(X) == 1:
        N.right = Node(X[0])
        N.left = buildTree(yindex, D)
    else:
        for i in range(yindex-1, -1, -1):
            V = D[i][0]
            if X == V:
                xindex = i
                break
        N.left = buildTree(yindex, D)
        N.right = buildTree(xindex, D)
    return N

def treeToXML(tree, fileList, outputFileName = None):
    #print("<tree height = \"" + str(tree.getLabel()) + "\">")
    if outputFileName is not None:
        try:
            file = open(outputFileName, 'w')
            file.write("<tree height = \"" + str(tree.getLabel()) + "\">\n")
        except IOerror as err:
            print("There was an issue with your output file for the xml tree")
        preorder(tree, tree, fileList, 0, file)
    #print("</tree>")
    if outputFileName is not None:
        file.write("</tree>")
        file.close()


def preorder(T, root, fileList, spaces, file = None):
    if(T is None):
        return
    left = T.getLeft()
    right = T.getRight()
    indent = " " * spaces
    if(left is not None and right is not None):
        if T != root:
            file.write(indent + "<node height =\""+ str(T.getLabel()) +"\">\n")
        preorder(left, root, fileList, spaces+4, file)
        preorder(right, root, fileList, spaces+4, file)
    else:
        data = fileList[T.getLabel()]
        file.write(indent + "<leaf height =\"0\" data =\"" + data +"\"/>\n")
    if left is not None and right is not None:
        if T != root:
            file.write(indent + "</node>\n")

def cosineSimilarity2(x1, x2):
    dotproduct = np.sum(np.multiply(x1, x2))
    magnitude_x1 = np.sqrt(np.sum(np.square(x1)))
    magnitude_x2 = np.sqrt(np.sum(np.square(x2)))
    return 1 - float(dotproduct)/ (magnitude_x1 * magnitude_x2)

def calculateDistancesCosine(D):
    dist_matrix=[] #define empty matrix
    n = len(D)
    for i in range(n):
        distances = np.zeros(n)
        dist_matrix.append(distances)
    for i in range(n):
        for j in range(i+1, n):
            x1 = D[i]
            x2 = D[j]
            dist = cosineSimilarity2(x1, x2)
            dist_matrix[i][j] = dist
    return dist_matrix

def getDistances(inputFile):
    file = open(inputFile, "r")
    lines = file.readlines()
    file.close()

    files = lines[0].replace("\n", "").split(",")
    lines = lines[1:]
    pixels = np.zeros((len(lines),3600))
    truth = []
    for i in range(len(lines)):
        line = lines[i].replace("\n", "")

        pixels[i] = [int(x) for x in line.split(',')[:-1]]
        truth.append(line.split(',')[-1])
    dist_matrix = calculateDistancesCosine(pixels)

    return dist_matrix, truth, files

def runHClustering(inputFile, dendogramFile):
    try:
        D, truth, fileList = getDistances(inputFile)
    except IOError as err:
        print("Your training set file was not found, or there was another issue with your file")
        return

    C, MIN_Clusters = agglomerative(D)
    root = buildTree(len(MIN_Clusters)-1,MIN_Clusters)
    treeToXML(root, fileList, dendogramFile)

runHClustering('data.csv', 'dendrogram.xml')
