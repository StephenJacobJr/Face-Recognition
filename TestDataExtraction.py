from DataReader import read_pgm
from ImageProcessor import getFeatures,genIndexTable,calcSingleFeature,getIntegralImage
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
    
    
def predict(classifiers,alphas,imgArray,indexTable):
    start = time.time()
    
#     imgFeatures = getFeatures(imgArray)
    integralImage = getIntegralImage(imgArray)
    end1 = time.time()
    product = []
    for i in xrange(len(classifiers)):
        c = classifiers[i]
        
#         featureValue = np.array([imgFeatures[c.feature]])
        featureValue = calcSingleFeature(indexTable,c.feature,integralImage)
        prediction = c.predict(featureValue,c.threshold,c.side) 
        prediction[prediction==-1] = 0
        product.append(prediction* alphas[i])
    result = np.array(product).sum()
    
#     print "Result:", result
#     print "Summed alphas:", 0.5*alphas.sum()
    
    end2 = time.time()
     
    print end1-start
    print end2-start
    print "-----------------------------"
    
    if result >= 0.5*alphas.sum():
#         print "I think this is a face."
        return 0
    else:
#         print "I think this is NOT a face."
        return 1

#524 0
#389 1

#3:11

if MUST_ADABOOST:
    if not MUST_GEN_FEATURES: features = np.load("features.npy")
    classifiers = adaboost(features,labels,weights,saveData)
else:
    classifierData = np.load("classifiers.npy")
    classifiers = loadClassifiers(classifierData)
    alphas = classifierData[:,3]
#     print classifierData


if MUST_SAVE_CLASSIFIERS:
    if not MUST_ADABOOST or not MUST_GEN_FEATURES: print "Must adaboost features to save classifiers."
    else:
        saveData = np.array(saveData)
        np.save("classifiers.npy",saveData)
        print "Data saved."
        
        
        
#TESTING

TEST_RANGE = 100

def testClassifier(classifiers,alphas):
    errors = 0.0
    indexTable = genIndexTable((19,19))
    for i in xrange(TEST_RANGE):
#         print i
        isFace = random.choice([FACE_INDEX,NON_FACE_INDEX])
#         if isFace == 1:
#             print "This is NOT a face."
#         else:
#             print "This IS a face."            
        
        fullPath = basePath[isFace]+"\\"+random.choice(dataset[isFace])
        try:
            imgArray = read_pgm(fullPath).astype(np.int64)
        except ValueError, e:
            print i
            continue
        
        result = predict(classifiers,alphas,imgArray,indexTable)
        if result != isFace:
            errors += 1
            
            
#         print "-------------------------------"
    print (errors/TEST_RANGE)*100,"% errors."

if MUST_RUN_TEST:
    testClassifier(classifiers,alphas)

# print "There are", len(classifiers), "classifiers."

