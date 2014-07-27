



import numpy as np
from math import log

LESSTHAN = 0
GREATERTHAN = 1

#http://fn.hgin.com/&_=1405041091
class DecisionStump:
    def __init__(self):
        self.threshold = 0
    def predict(self,data,threshold,side):
        result = np.ones(data.shape)
        if side == GREATERTHAN:
            thres = data <= threshold #array of booleans
        else:
            thres = data > threshold
        result[thres] = -1 # changes the indexes where thres is True to -1, false stay the same
        return result
    def train(self,data,labels,weights):
        minError = 100000
        numstep = 10 #number of threshold tries
        numfeat = data.shape[0]
        numexamples = data.shape[1]
        for feature in xrange(numfeat):
            maxi = data[feature,:].max()
            mini = data[feature,:].min()
            for step in xrange(-1,numstep+1):
                threshold = step*(maxi-mini)/numstep +mini
                for side in [LESSTHAN,GREATERTHAN]:                    
                    prediction = self.predict(data[feature,:],threshold,side)
                    error = np.ones(numexamples)
                    negL = labels[labels == -1]
                    negP = prediction[labels == -1]
                    gotRight = prediction == labels
                    error[gotRight] = 0
                    weightedError = np.dot(weights,error)
                    if weightedError < minError:
                        minError = weightedError
                        self.error = minError
                        self.feature = feature
                        self.prediction = prediction.copy()
                        self.threshold = threshold
                        self.side = side
        
class FinalClassifier:
    def __init__(self,classifiers, alphas=None):
        thresholds = []
        sides = []
        features = []
        alphaFlag = True
        if alphas == None:
            alphas = []
            alphaFlag = False
        for c in classifiers:
            if not alphaFlag:
                alpha = float( 0.5 * log( (1.0-c.error) / max(c.error, 1e-16) ) )
                alphas.append(alpha)
            features.append(c.feature)
            thresholds.append(c.threshold)
            sides.append(c.side)
            
        self.featIndexes = np.array(features)
#         print self.featIndexes
        self.thresholds = np.array(thresholds)
        self.sides = np.array(sides)
        self.sides[self.sides==0] = -1
        self.alphas = np.array(alphas)
    def predict(self,features):
        selFeat = features[self.featIndexes]
#         print selFeat
#         print selFeat
        predictions = np.ones(selFeat.shape[0])
        
#         print self.sides
        thresholded = self.sides*selFeat <= self.sides*self.thresholds #Problem with the equal, but not that big a deal

#         print thresholded
        predictions[thresholded] = 0
#         print predictions
        result = (predictions*self.alphas).sum()
#         print result
#         print "RESULT:",result
        return result < 0.5*self.alphas.sum()
    def test(self, features, labels):
        
        trueNegRate = 0.0
        falsePosRate = 0.0
        error = 0.0
        for i in xrange(features.shape[1]): #transposed to iterate over columns
            imgFeatures = features.T[i]
            face = labels[i]
            
            prediction = self.predict(imgFeatures)
            
            #Update True negative rate
            if labels[i] == 1 and prediction == 1:
                trueNegRate += 1
                    
            #Update false positive rate
            if labels[i] != 0 and prediction == 0:
                falsePosRate += 1
            if labels[i]!=prediction:
                error += 1
        trueNegRate = trueNegRate*100/(features.shape[1]/2)
        falsePosRate = falsePosRate*100/(features.shape[1]/2)
        error = error*100/features.shape[1]
        print "True negative:",  trueNegRate,"%"
        print "False positive:",falsePosRate,"%"
        print "Error:",error,"%"
        print "--------------------"
        return (trueNegRate,falsePosRate)
            
            
def loadClassifiers(path):
    classifierData = np.load(path)
    alphas = classifierData[:,3]
    classifiers = []    
    for i in xrange(classifierData.shape[0]):
        classifier = DecisionStump()
        classifier.threshold = classifierData[i,0]
        classifier.feature = int(classifierData[i,1])
        classifier.side = classifierData[i,2]
        classifiers.append(classifier)
    return classifiers,alphas
    
    
def saveClassifiers(classifiers,path):    
    saveData = []
    for DS in classifiers:
        alpha = float( 0.5 * log( (1.0-DS.error) / max(DS.error, 1e-16) ) )
        saveData.append((DS.threshold, DS.feature, DS.side, alpha))
    np.save(path,saveData)
    