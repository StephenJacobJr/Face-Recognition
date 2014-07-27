from DataReader import read_pgm
from ImageProcessor import genIndexTable,getIntegralImage,genFeatureSet,calcImgData
from DecisionStump import *
from math import log
import numpy as np
import os
import time
import random

#DATA LOADING AND FEATURE EXTRACTING ###############################################################


FEATURE_NUMBER = 63960
IMG_TRAIN_NUMBER = 1000
FACE_INDEX = 0
NON_FACE_INDEX = 1

MUST_GEN_FEATURES = False #IF NOT GENED AND MUST ADABOOST: WILL BE LOADED
MUST_ADABOOST = False
MUST_SAVE_CLASSIFIERS = False
MUST_RUN_TEST = True

basePath = ["Faces\\train\\face","Faces\\train\\non-face"]

faces = os.listdir(basePath[FACE_INDEX])
nonFaces = os.listdir(basePath[NON_FACE_INDEX])

dataset = [faces,nonFaces]

# print dataset[0][524]
# print dataset[1][389]

print "There are", len(faces), "faces."
print "There are", len(nonFaces), "non-faces."


labels = np.ones(IMG_TRAIN_NUMBER*2)

weights = np.ones( (1,IMG_TRAIN_NUMBER*2) )/ (IMG_TRAIN_NUMBER*2)

def genFeatures(labels):
    features = []
    for i in xrange(FEATURE_NUMBER): #To be able to use np.append we need the initial array to have the right shape
        features.append([])
    features = np.array(features)
    
    
    for imgType in [FACE_INDEX,NON_FACE_INDEX]: 
        for imgIndex in xrange(IMG_TRAIN_NUMBER):
            
            print "Image number", IMG_TRAIN_NUMBER*imgType + imgIndex
            
            fullPath = basePath[imgType]+"\\"+dataset[imgType][imgIndex]
            try:
                imgArray = read_pgm(fullPath).astype(np.int64)
            except ValueError, e:
                print imgIndex,imgType
                break
            
            imgFeatures = getFeatures(imgArray)
    #         print "The feature array of this particular image has the following shape:", imgFeatures.shape
            features = np.append(features, imgFeatures, axis=1)
            
            if imgType == NON_FACE_INDEX:
                labels[IMG_TRAIN_NUMBER + imgIndex] = -1
            
    print "The feature array has the following shape:", features.shape
    return features


    

if MUST_GEN_FEATURES:
    features = genFeatures(labels)
    np.save("features.npy",features)

#ADABOOSTING###########################################################


NUM_FEATURES_KEPT = 200


saveData = []
    
def adaboost(features,labels,weights,saveData):
    classifiers = []
    
    print features.shape
    print labels
    totalTime = 0
    for stumpIndex in xrange(NUM_FEATURES_KEPT):
        
        start = time.time()
        
        print "Training feature number", stumpIndex, "..."
#         print "Average training time:", totalTime/(stumpIndex+1), "seconds."
        DS = DecisionStump()
        DS.train(features, labels, weights)
#         print "Current threshold is:", DS.threshold
#         print "Current feature is:", DS.feature
        classifiers.append(DS)
        
        
        error = DS.error
        print error
        alpha = float( 0.5 * log( (1.0-error) / max(error, 1e-16) ) )
        weights *= np.exp( -labels*alpha*DS.prediction )
        weights /= weights.sum()
            
        saveData.append((DS.threshold, DS.feature, DS.side, alpha))
        
        end = time.time()
        totalTime += end-start
    return classifiers

def loadClassifiers(classifierData):
    classifiers = []    
    for i in xrange(classifierData.shape[0]):
        classifier = DecisionStump()
        classifier.threshold = classifierData[i,0]
        classifier.feature = int(classifierData[i,1])
        classifier.side = classifierData[i,2]
        classifiers.append(classifier)
    return classifiers
    
    
    

    
    
def predict(features,sides,thresholds,alphas,imgArray):
    
#     print features
    predictions = np.ones(features.shape[0])
    
    
    thresholded = sides*features <= sides*thresholds #Problem with the equal, but not that big a deal
#     print thresholded
    predictions[thresholded] = 0
#     print "-------------------------"
#     print sides
#     print thresholded
    
    result = (predictions*alphas).sum()
#     print "RESULT:",result
    return result < 0.5*alphas.sum()
    
    

if MUST_ADABOOST:
    if not MUST_GEN_FEATURES: 
        features = np.load("features.npy")
        labels[range(1000,2000)] = -1
    classifiers = adaboost(features,labels,weights,saveData)
else:
    classifierData = np.load("classifiers.npy")
    classifiers = loadClassifiers(classifierData)
#     alphas = classifierData[:,3]
#     print classifierData


if MUST_SAVE_CLASSIFIERS:
    if not MUST_ADABOOST or not MUST_GEN_FEATURES: print "Must adaboost features to save classifiers."
    else:
        saveData = np.array(saveData)
        np.save("classifiers.npy",saveData)
        print "Data saved."
        
        
        
#TESTING

TEST_RANGE = 100
FEATURE_NUMBER = 5

def sortByFeature(array,featTypes):
    
    featSorted = []
    for i in xrange(FEATURE_NUMBER):
        featSorted.append(array[featTypes==i])
    
    return np.concatenate(featSorted)


def testClassifier(classifierData,classifiers):
    errors = 0.0
    alphas = classifierData[:,3]    
    thresholds = classifierData[:,0]
    sides = classifierData[:,2]
    sides[sides==0] = -1
    
    featIndexes = classifierData[:,1].astype(np.int64)
    indexes = featIndexes
    t0 = 0
    t1 = 0
    long = 0
    for i in xrange(1001,1002):
#         print i
#         isFace = random.choice([FACE_INDEX,NON_FACE_INDEX])
        isFace = 1
#         if isFace == 1:
#             print "This is NOT a face."
#         else:
#             print "This IS a face."            
        
        fullPath = basePath[isFace]+"\\"+random.choice(dataset[isFace])
        print fullPath
        try:
            imgArray = read_pgm(fullPath).astype(np.int64)
        except ValueError, e:
            print "ValueError:",i
            print e
            continue     
         
        indexTable = genIndexTable(imgArray.shape)
        start = time.time()
#         features = genFeatureSet(classifierData[:,1].astype(np.int64),imgArray, indexTable, integralImage)
        data = calcImgData(featIndexes, imgArray, indexTable)
        features = genFeatureSet(featIndexes.shape, *data)
        end1 = time.time()
#         featTypes = indexTable[featIndexes,0]
#         sides = sortByFeature(sides,featTypes)
#         thresholds = sortByFeature(thresholds,featTypes)
#         alphas = sortByFeature(alphas,featTypes)
        result = predict(features,sides,thresholds,alphas,imgArray)
        end2 = time.time()
        
#         if isFace == 0:
#             print "This is a face"
#         else:
#             print "This is not a face."
#             
#         if result == 0:
#             print "Adaboost thinks this is a face."
#         else:
#             print "Adaboost thinks this is not a face."
#         print "-----------------------------------------"
        if result != isFace:
            errors += 1
#         print end1-start
#         if end1-start != 0.0:
#             long += 1
#             print end1 - start
#             print fullPath
#             print fullPath
        t0 += end1-start
        t1 += end2-end1
        
    print "LONG:",long
    print t0/TEST_RANGE
#     print t1/TEST_RANGE
#     print "-------------------------------"
    print (errors/TEST_RANGE)*100,"% errors."


def test_slow(classifierData,classifiers):
    errors = 0.0  
    t0 = 0
    long = 0    
    
    #LOAD PICTURE
    path = "Faces\\train\\face\\face01675.pgm"
    imgArray = read_pgm(path).astype(np.int64)
    #############################################
        
    featIndexes = classifierData[:,1].astype(np.int64)
    #LOAD DATA FOR QUICK FEATURE CALCULATION
    featShape = featIndexes.shape
    indexTable = genIndexTable(imgArray.shape)
    data = calcImgData(featIndexes, imgArray, indexTable)
    
    integralImage = data[0]
    featSortedIndexes = data[1]
    X = data[2]
    Y = data[3]
    ENDX = data[4]
    ENDY = data[5]
    MIDDLEX = data[6]
    MIDDLEY = data[7]
    THIRDX1 = data[8]
    THIRDX2 = data[9]
    THIRDY1 = data[10]
    THIRDY2 = data[11]
    ########################################
    
    for i in xrange(10000):
        start = time.time()
        features = genFeatureSet(featShape, featSortedIndexes, X, Y, ENDX, ENDY, MIDDLEX, MIDDLEY, THIRDX1, THIRDX2, THIRDY1, THIRDY2, integralImage)
        end = time.time()
        
        t0 += end-start
        if end-start!=0:
#             print end-start
            long += 1
    print t0/10000
    print long
           

def testRange(classifierData, classifiers, dataRange):
    alphas = classifierData[:,3]    
    thresholds = classifierData[:,0]
    sides = classifierData[:,2]
    sides[sides==0] = -1
    
    featIndexes = classifierData[:,1].astype(np.int64)
    for isFace in xrange(1):
        for i in xrange(1000,1001):
            
            fullPath = basePath[isFace]+"\\"+dataset[isFace][i]
            print fullPath
            try:
                imgArray = read_pgm(fullPath).astype(np.int64)
            except ValueError, e:
                print "ValueError:",i
                print e
                continue     
             
            indexTable = genIndexTable(imgArray.shape)
            data = calcImgData(featIndexes, imgArray, indexTable)
            features = genFeatureSet(featIndexes.shape, *data)
            result = predict(features,sides,thresholds,alphas,imgArray)
    

if MUST_RUN_TEST:
#     test_slow(classifierData,classifiers)
#     testClassifier(classifierData,classifiers)
    testRange(classifierData, classifiers, 0)

    
    
    
# print "There are", len(classifiers), "classifiers."

