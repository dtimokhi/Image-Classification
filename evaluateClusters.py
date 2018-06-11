import time
import xml.etree.ElementTree as ET
import numpy as np
import math
import pandas as pd
import random
import sys
import os

def printRecur(root, cluster_roots, outliers, truth):
    root_index = 0
    while len(cluster_roots) < 2:
        cluster_roots.remove(root)
        for elem in root:
            if elem.tag == 'node':
                leaves = recursivePrint(elem, {}, truth)
                count = 0
                for key in leaves.keys():
                    count += leaves[key]
                if count>10:
                    cluster_roots.append(elem)
                else:
                    for key in leaves:
                        for j in range(leaves[key]):
                            outliers.append(key)
        root = np.argmax(np.array([e.attrib['height'] for e in cluster_roots]))
        root = cluster_roots[root]
    return cluster_roots, outliers

def recursivePrint(root, leaves, truth):
    for elem in root.getchildren():
        if elem.tag == 'leaf':
            author = truth[elem.get('data')]
            if author in leaves:
                leaves[author] += 1
            else:
                leaves[author] = 1
        else:
            recursivePrint(elem, leaves, truth)
    return leaves


def evaluateClusters(groundTruthFile, dendogramFile):
    try:
        file = open(groundTruthFile)
        truth = {}
        fileList = []
        for line in file.readlines():
            words = line.split(",")
            fileList.append(words[0].strip())
            truth[words[0].strip()] = words[1].strip()
    except IOError as err:
        print("Your ground truth file was not found, or there was another issue with your file")
        return
    try:
        tree = ET.parse(dendogramFile)
        root = tree.getroot()
        clusters2, outliers = printRecur(root, [root], [], truth)

    except IOError as err:
        print("Your dendogram file was not found, or there was another issue with your file")
    i = 0
    authors = {}
    clusters = []
    for root in clusters2:
        count = 0
        leaves = recursivePrint(root, {}, truth)
        for key in leaves.keys():
            count += leaves[key]
        values = []
        plurality = max(leaves, key=lambda k: leaves[k])
        for key, value in sorted(leaves.items(), key=lambda x: x[1]*-1):
            if key not in authors:
                authors[key] = [0,0]
            if key == plurality:
                authors[key][0] += value
            else:
                authors[key][1] += value
            for j in range(value):
                values.append(key)
        clusters.append((float(leaves[plurality])/count, count, plurality, values))
    for key in outliers:
        if key not in authors:
            authors[key] = [0,0]
    clusters = sorted(clusters, reverse = True)
    i = 0
    for purity, count, plurality, values in clusters:
        i += 1
        print("")
        print("group " + str(i) + ":  size " + str(count))
        print("Plurality Face: " + plurality + "   purity = " + str(purity))
        print("")
        for value in values:
            print(value)
    print("")
    print("outliers: size " +str(len(outliers)))
    for outlier in outliers:
        print(outlier)

    print("")
    print("")
    print("Accuracy of detecting each face")
    sortedAuthors = sorted(list(authors.keys()))
    for key in sortedAuthors:
        correct = authors[key][0]
        incorrect = authors[key][1]
        precision = float(correct)/(correct+incorrect)
        recall = float(correct)/(len(truth)/2)
        print(key + ":")
        print("           precision = " + str(round(precision,4)) + "   recall = " + str(round(recall,4)))

evaluateClusters('groundTruth.csv', 'dendrogram.xml')
