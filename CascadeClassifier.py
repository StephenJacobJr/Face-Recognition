from DecisionStump import *
import numpy as np
import os
from ImageReader import read_pgm
from ImageProcessor import genIndexTable,genFeatureSet,calcImgData
from math import log 





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




def genStage(features, labels, validationFeatures, vLabels, target, trainingRange, validationRange):
    #target: boolean: negative rate, false pos rate
    
    
    weights = np.ones( (1,len(trainingRange)*2) )/ (len(trainingRange)*2)
    negRate,falsePosRate = -1,101
    
     
    classifiers = []
    while negRate <= target[0] or falsePosRate >= target[1]:
        DS = DecisionStump()
        DS.train(features, labels, weights)
        classifiers.append(DS)        
        error = DS.error
        alpha = float( 0.5 * log( (1.0-error) / max(error, 1e-16) ) )
        weights *= np.exp( -labels*alpha*DS.prediction )
        weights /= weights.sum()
        print "Number of classifiers:", len(classifiers)
        print "Feature selected:",DS.feature
        FC = FinalClassifier(classifiers)
        negRate,falsePosRate = FC.test(validationFeatures, vLabels)
#         print negRate, falsePosRate
    print negRate, target[0]
    print falsePosRate, target[1]
    return FinalClassifier(classifiers)
        


# features = genFeatures(PATH, range(IMG_TRAIN_NUMBER), FEATURE_NUMBER, IMG_SHAPE)
# features = genFeatures(PATH, range(1000,2000), FEATURE_NUMBER, IMG_SHAPE)
# np.save("validationFeatures.npy",features)
# vFeatures = vFeatures[:,[0,1]]
# print vFeatures.shape


#TEST TRAIN
features = np.load("features.npy")
vFeatures = np.load("validationFeatures.npy")
labels = np.ones(2000)
labels[range(1000,2000)] = -1
vLabels = np.ones(2000)
vLabels[range(1000)] = 0
genStage(features, labels, vFeatures, vLabels, (100,40), range(1000),0)
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
