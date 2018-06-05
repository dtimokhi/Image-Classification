import numpy as np
import math
import pandas as pd
import random
import sys
import os
import string

def diskKMeans(D, k, fileList):
    m = SelectInitialCentroidsRandomly(D, k)
    stoppingCondition = False
    previousClusters = []
    while stoppingCondition == False:
        s = []
        num = []
        cl = []
        clFiles = []
        for j in range(k):
            s.append([0 for i in range(D.shape[1])])
            num.append(0)
            cl.append([])
            clFiles.append([])
        for i in range(D.shape[0]):
            min = 0
            minDistance = cosineSimilarity(D[0], m[j])
            for j in range(k):
                x = D[i]
                newDistance = cosineSimilarity(x, m[j])
                if minDistance > newDistance:
                    min = j
                    minDistance = newDistance
            cl[min].append(x)
            clFiles[min].append(fileList[i])
            for v in range(D.shape[1]):
                s[min][v] += x[v]
            num[min] += 1
        for i in range(k):
            cl[i] = np.array(cl[i])
        s = np.array(s)
        num = np.array(num)
        for i in range(k):
            averages = []
            if num[i] == 0:
                m[i] = np.array([0.0 for i in range(D.shape[1])])
            else:
                m[i] = np.divide(s[i], num[i])
        cl = np.array(cl)
        stoppingCondition = isStoppingCondition(previousClusters, cl)
        previousClusters = cl
    return cl, m, clFiles

#if there is no reassignment of points between clusters then stop
def isStoppingCondition(previousClusters, cl):
    if len(previousClusters) == 0:
        return False
    #for cluster i in clusters
    for i in range(len(cl)):
        #for each element in the cluster
        current = cl[i]
        previous = previousClusters[i]
        #print(current)
        for c in current:
            if c not in previous:
                return False
    return True

#x1 and x2 are arrays (of the same size) of values
def eucledianDist(x1, x2):
    return np.sum(np.square(np.subtract(x1,x2)))

def cosineSimilarity(x1, x2):
    dotproduct = np.sum(np.multiply(x1,x2))
    magnitude_x1 = np.sqrt(np.sum(np.square(x1)))
    magnitude_x2 = np.sqrt(np.sum(np.square(x2)))
    if float(dotproduct)/ (magnitude_x1 * magnitude_x2) < 0:
        print("ISSUE WITH COSINE SIMILARITY: GOT NEGATIVE VALUE")
    return 1- float(dotproduct)/ (magnitude_x1 * magnitude_x2)

def SelectInitialCentroidsRandomly(D, k):
    nums = range(D.shape[0])
    choices = random.sample(nums, k)
    centroids = [D[choice] for choice in choices]
    return centroids

def SelectInitialCentroids2(D, k):
    count = 0
    sums = [0 for i in range(D.shape[1])]
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            sums[j] += D[i,j]
            count += 1
    sums = np.array(sums)
    overallCentroid = np.divide(sums,count)
    c1 = D[0]
    c1Dist = cosineSimilarity(c1, overallCentroid)
    centroidIndicies = [0]
    for i in range(D.shape[0]):
        row = D[i]
        distance = cosineSimilarity(row, overallCentroid)
        if distance > c1Dist:
            c1 = row
            c1Dist = distance
            centroidIndicies[0] = i
    centroids = [c1]
    for i in range(1, k):
        potentialIndicies = [x for x in range(D.shape[0]) if x not in centroidIndicies]
        m1 = centroids[i-1]
        m2 = D[potentialIndicies[0]]
        m2Dist = np.sum(np.array([cosineSimilarity(x, m2) for x in centroids]))
        centroidIndicies.append(potentialIndicies[0])
        for j in potentialIndicies:
            row = D[j]
            distance = np.sum(np.array([cosineSimilarity(x, row) for x in centroids]))
            if distance > m2Dist:
                m2 = row
                m2Dist = distance
                centroidIndicies[i] = j
        centroids.append(m2)
    return centroids

def getClusterData(clusters, centroids, fileList):
    print("")
    for i in range(len(clusters)):
        cluster = clusters[i]
        center = centroids[i]
        fileName = fileList[i]
        print("Cluster " + str(i+1) + ":")
        line2 = "Center: "
        for c in center:
            line2 += str(c) + ", "
        print(line2[:-2])
        if len(cluster) > 0:
            maxDistToCenter = cosineSimilarity(center, cluster[0])
            minDistanceToCenter = cosineSimilarity(center, cluster[0])
            sum = 0
            count = 0
            for c in cluster:
                newDist = cosineSimilarity(center, c)
                if newDist > maxDistToCenter:
                    maxDistToCenter = newDist
                if newDist < minDistanceToCenter:
                    minDistanceToCenter = newDist
                count += 1
                sum += newDist
            print("Max Dist. to Center: " + str(maxDistToCenter))
            print("Min Dist. to Center: " + str(minDistanceToCenter))
            print("Avg Dist. to Center: " + str(sum/float(count)))
        print(str(len(cluster)) + " Points:")
        for c in cluster:
            print(c)
        print("")


def getDataSet(inputFile):
    file = open(inputFile, "r")
    lines = file.readlines()
    file.close()
    files = lines[0].replace("\n", "").split(",")
    lines = lines[1:]
    pixels = np.zeros((len(lines),3600))
    truth = {}
    for i in range(len(lines)):
        line = lines[i].replace("\n", "")
        pixels[i] = [int(x) for x in line.split(',')[:-1]]
        truth[files[i]] = line.split(',')[-1]
    return pixels, files, truth

def runKmeans(inputFile, k):
    try:
        df, files, truth = getDataSet(inputFile)
    except IOError as err:
        print("Your training set file was not found, or there was another issue with your file")
        return
    if k.__class__.__name__ != 'int' or k<=1 or k>df.shape[0]:
        print("Your value for k must be an integer, greater than 1, and less than the size of your data")
        return
    clusters, centroids, clFiles = diskKMeans(df, k, files)
    return clFiles, truth


# if len(sys.argv) < 3:
#     print("Be sure to specify a file name and k")
# elif len(sys.argv) == 3:
#     runKmeans(sys.argv[1], int(sys.argv[2]))
# elif len(sys.argv) == 4:
#     results, numCols = runKmeans(sys.argv[1], int(sys.argv[2]))
#     try:
#         file = open(sys.argv[3], "w")
#         line = ""
#         for i in range(numCols):
#             line += "col" + str(i+1) + ","
#         line += "group"
#         file.write(line)
#         for i in range(len(results)):
#             for j in range(len(results[i])):
#                 line = "\n"
#                 for num in results[i][j]:
#                     line += str(num) + ","
#                 line += "group " + str(i+1)
#                 file.write(line)
#         file.close()
#     except IOError as err:
#         print("There was an issue with your output file")

# fileList = []
# vectorList = []
#
# file = open("outputLongStop.csv", "r")
# lines = file.readlines()
# file.close()
# for line in lines:
#     words = line.replace("\n", "").split(",")
#     fileList.append(words[0])
#     vectorList.append(words[1:])
#
# fileList = fileList[1:10]
# vectorList = vectorList[1:10]
# df = pd.DataFrame(vectorList)

clusters, truth = runKmeans("data.csv", 2)
for i in range(len(clusters)):
    print("Cluster " + str(i+1))
    for c in clusters[i]:
        print (truth[c])
