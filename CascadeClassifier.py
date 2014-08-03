from DecisionStump import *
import numpy as np
import os
from ImageReader import read_pgm
from ImageProcessor import genIndexTable,genFeatureSet,calcImgData
from math import log 


import time


IMG_TRAIN_NUMBER = 1000
FEATURE_NUMBER = 63960
IMG_SHAPE = (19,19)
PATH = ["Faces\\train\\face","Faces\\train\\non-face"]


def genFeatures(path, trainingRange, numberOfFeatures, imgShape):
    
    features = np.empty((numberOfFeatures, 0))
    
    faces = os.listdir(path[0])
    nonFaces = os.listdir(path[1])
    dataset = [faces, nonFaces]
    
    featureIndexes = np.arange(numberOfFeatures)
    indexTable = genIndexTable(imgShape)
    
    labels = np.ones(len(trainingRange)*2)    
    
    for imgType in xrange(2): #face or non face 
        for imgIndex in trainingRange:
            
            print "Image number", len(trainingRange)*imgType + imgIndex -trainingRange[0]
            
            #LOAD IMAGE
            fullPath = path[imgType]+"\\"+dataset[imgType][imgIndex]
            print fullPath
            try: imgArray = read_pgm(fullPath).astype(np.int64)
            except ValueError, e: print imgIndex,imgType; break
            #############
            
            #GEN FEATURES
            data = calcImgData(featureIndexes, imgArray, indexTable)
            imgFeatures = genFeatureSet(numberOfFeatures, *data)
            features = np.append(features, np.expand_dims(imgFeatures,1), axis=1)
            #############
            
            if imgType == 1:
                labels[len(trainingRange) + imgIndex - trainingRange[0]] = -1
            
    return features




def genStage(features, labels, validationFeatures, vLabels, target, trainingRange):
    #target: boolean: negative rate, false pos rate
    
    
    allWeights = np.ones( (1,len(trainingRange)*2) )/ (len(trainingRange)*2)
    truePosRate,falsePosRate = -1,101
    
    allFeatures = features.copy()
    allLabels = labels.copy()
    featNumb = 1
    
     
    classifiers = []
    while truePosRate < target[0] or falsePosRate > target[1]:
        print featNumb
        
        features = allFeatures[:featNumb]
        labels = allLabels
        weights = allWeights[:featNumb] 
        
        
        
        DS = DecisionStump()
        DS.train(features, labels, weights)
        classifiers.append(DS)        
        error = DS.error
        alpha = float( 0.5 * log( (1.0-error) / max(error, 1e-16) ) )
        weights *= np.exp( -labels*alpha*DS.prediction )
        weights /= weights.sum()
#         print "Number of classifiers:", len(classifiers)
#         print "Feature selected:",DS.feature
        
        
        FC = FinalClassifier(classifiers)
        truePosRate,falsePosRate = FC.test(validationFeatures, vLabels) 
       
        print truePosRate, falsePosRate
        step = 0
        maxi = features[DS.feature,:].max()
        mini = features[DS.feature,:].min()
        while truePosRate < target[0]:        
            DS.threshold -= step*DS.side*(maxi-mini)/100
        
            FC = FinalClassifier(classifiers)
            truePosRate,falsePosRate = FC.test(validationFeatures, vLabels) 
               
            print "\t",truePosRate,falsePosRate
#             print "\t",truePosRate < target[0] 
            if falsePosRate > target[1] or step>40: break
            step += 1
            
            
            
        featNumb += 1000
        
        classifiers = []
#         print negRate, falsePosRate
    print truePosRate, target[0]
    print falsePosRate, target[1]
    return FinalClassifier(classifiers)
        

def genStage2(features, labels, validationFeatures, vLabels, falsePosTarget, truePosTarget, trainingRange):
    "Generates a stage of the cascade classifier with the targeted rate of detection and false positives."
    
    falsePosResult = 100.0 # %
    FC = None
    
    stumps = []    
    weights = np.ones( (1,len(trainingRange)*2) )/ (len(trainingRange)*2)
    
    #number of classifiers
    i = 0
    
    while falsePosResult > falsePosTarget:
        
        print "Generating classifier number", i
        
        #create classifier
        stump = DecisionStump()
        stump.train(features,labels,weights)
        
        #adaboost update    
        alpha = float( 0.5 * log( (1.0-stump.error) / max(stump.error, 1e-16) ) )
        weights *= np.exp( -labels*alpha*stump.prediction )
        weights /= weights.sum()
        #add to classifiers
        stumps.append(stump)
        
        
        #generate final classifier
        FC = FinalClassifier(stumps)
        
        #Get actual detection and falsePos rates from validation set
        truePosResult, falsePosResult = FC.test(validationFeatures, vLabels)
        print truePosResult, falsePosResult
        
        #lower threshold of classifier until we get acceptable detection rate
        lastTruePos = truePosResult
        while truePosResult < truePosTarget:
            
            #Lowering the threshold increases both the true positives and the false positives.
            if stump.side: # greaterThan
                stump.threshold -= 0.1
            else: # lesserThan
                stump.threshold += 0.1
            
            
            #generate final classifier
            FC = FinalClassifier(stumps)
            
            #Get actual detection and falsePos rates from validation set
            truePosResult, falsePosResult = FC.test(validationFeatures, vLabels)
            
            if truePosResult != lastTruePos:
                print truePosResult, falsePosResult
                lastTruePos = truePosResult
        
        i+= 1
        
    return FC
    
    

# features = genFeatures(PATH, range(IMG_TRAIN_NUMBER), FEATURE_NUMBER, IMG_SHAPE)
# features = genFeatures(PATH, range(1000,2000), FEATURE_NUMBER, IMG_SHAPE)
# np.save("validationFeatures.npy",features)
# vFeatures = vFeatures[:,[0,1]]
# print vFeatures.shape


#TEST TRAIN
# features = np.load("features.npy")
# vFeatures = np.load("validationFeatures.npy")
# labels = np.ones(2000)
# labels[range(1000,2000)] = -1
# vLabels = np.ones(2000)
# vLabels[range(1000)] = 0
# genStage(features, labels, vFeatures, vLabels, (100,40), range(1000),0)
##################################
#TEST TRAIN 2
features = np.load("features.npy")
vFeatures = np.load("validationFeatures.npy")
labels = np.ones(2000)
labels[range(1000,2000)] = -1
vLabels = np.ones(2000)
vLabels[range(1000)] = 0
genStage2(features, labels, vFeatures, vLabels, 40.0, 100.0, range(1000))
##################################
 
#TEST ALREADY TRAINED
# classifiers, alphas = loadClassifiers("classifiers.npy")
# FC = FinalClassifier(classifiers, alphas)
# vFeatures = np.load("validationFeatures.npy")
# vLabels = np.ones(2000)
# vLabels[range(1000)] = 0
# negRate,falsePosRate = FC.test(vFeatures, vLabels)
# print negRate, falsePosRate
##############################
# 
# features = genFeatures(PATH,range(1000,1001), FEATURE_NUMBER, IMG_SHAPE)
# FC.predict(features.T[0])
