



import numpy as np


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
#                     print "Prediction:", prediction
                    error = np.ones(numexamples)
#                     print "Error:", error
                    gotRight = prediction == labels
#                     print "Got right:", gotRight
                    error[gotRight] = 0
                    
                    
#                     print "Weights:", weights
#                     print "Error:", error
                    weightedError = np.dot(weights,error)
                    #print "Weighted Error", weightedError
                    if weightedError < minError:
                        minError = weightedError
                        self.error = minError
                        self.feature = feature
                        self.prediction = prediction.copy()
                        self.threshold = threshold
                        self.side = side
        


