import time
import numpy as np
import math
import pandas as pd
import random
import sys
import os
import string

def cosineSimilarity2(x1, x2):
    dotproduct = np.sum(np.multiply(x1, x2))
    magnitude_x1 = np.sqrt(np.sum(np.square(x1)))
    magnitude_x2 = np.sqrt(np.sum(np.square(x2)))
    return float(dotproduct)/ (magnitude_x1 * magnitude_x2)

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

def cosineSimilarity(x1, x2):
    dotproduct = sum([x1[i]*x2[i] for i in range(len(x1))])
    magnitude_x1 = math.sqrt(sum([x**2 for x in x1]))
    magnitude_x2 = math.sqrt(sum([x**2 for x in x2]))
    return float(dotproduct)/ (magnitude_x1 * magnitude_x2)

def KNN(D, k, truth, files, x, outputFile = None):
    distances = []
    for i in range(len(D)):
        if i < x:
            dist = D[i][x]
            distances.append((dist, i))
        elif i > x:
            dist = D[x][i]
            distances.append((dist, i))
    distances = sorted(distances, reverse = True)
    count_neighbors = {}
    for i in range(k):
        face = truth[distances[i][1]]
        if face not in count_neighbors:
            count_neighbors[face] = 0
        count_neighbors[face] += 1
    plurality_class = max(count_neighbors, key=lambda k: count_neighbors[k])
    return files[x], plurality_class, truth[x]

def loopKNN(dist_matrix, k, truth, files, outputFile = None):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    incorrect = []
    for x in range(len(dist_matrix)):
        f, p, t = KNN(dist_matrix, k, truth, files, x)
        if p == t:
            if t == 'smile':
                TP += 1
            else:
                TN += 1
        else:
            if p == 'smile':
                FP += 1
            else:
                FN += 1
            incorrect.append(t + ": " + f)
        if outputFile is None:
            print(f + " --> predicted: " + p + "  actual: " + t)
        else:
            outputFile.write(f + "," + p + "," + t + "\n")
    print("")
    print("Confusion Matrix: ")
    print("             Frown          Smile")
    print("Frown         " + str(TN) + "            " + str(FP))
    print("Smile         " + str(FN) + "             " + str(TP))
    print("")
    print("Accuracy: "+str(float(TP+TN)/(TP+TN+FP+FN)))
    print("Precision: " + str(float(TP)/(TP+FP)))
    print("Recall: " + str(float(TP)/(TP+FN)))
    print("")
    print("Images Incorrectly Predicted: ")
    for i in incorrect:
        print(i)


def runKNN(inputFile, k, outputFile = None):
    try:
        dist_matrix, truth, files = getDistances(inputFile)
    except IOError as err:
        print("Your training set file was not found, or there was another issue with your file")
        return
    if k < 1:
        print("Your value for k must be at least 1")
        return
    if outputFile is not None:
        output = open(outputFile, "w")
        for x in range(len(dist_matrix)):
            KNN(dist_matrix, k, truth, files, x, output)
            if x != len(dist_matrix)-1:
                output.write("\n")
        output.close()
    else:
        loopKNN(dist_matrix, k, truth, files)

runKNN('data.csv', 5)
