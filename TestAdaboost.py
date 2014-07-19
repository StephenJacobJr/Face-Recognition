import numpy as np
import matplotlib.pyplot as plt
from DecisionStump import *
from math import log,exp
import random
 
 
 
 
def genCircleInRect(data,result):
    for i in xrange(1000):
        x = 1000
        y = 1000
        while ( ( x**2+y**2) >= 4):
            x = random.uniform(-10,10)
            y = random.uniform(-10,10)
        data[0].append(x)
        data[1].append(y)
        result.append(-1)
    for i in xrange(5000):
        x = 0.6
        y = 0.3
        while (x-0.6)**2 + (y-0.3)**2 < 0.2**2 or ( x**2+y**2) <= 4:
            x = random.uniform(-10,10)
            y = random.uniform(-10,10)
        data[0].append(x)
        data[1].append(y)
        result.append(1)            
    return data,result
 
def genDiagonalInRect(data,result):
    for i in xrange(1000):
         
        x = random.uniform(-10,10)
        y = x + random.uniform(-1,1)
        data[0].append(x)
        data[1].append(y)
        result.append(-1)
    for i in xrange(5000):
        x = 0.6
        y = 0.3
        while (x-0.6)**2 + (y-0.3)**2 < 0.2**2 or ( x==y):
            x = random.uniform(-10,10)
            y = random.uniform(-10,10)
        data[0].append(x)
        data[1].append(y)
        result.append(1)            
    return data,result
 
     
     
     
x = np.array([[2,2,3,4,4,3.1,2.9,2.9,3.0,3.0,2,4],
              [2,4,3,2,4,3.1,2.9,3.0,4.0,2.0,3,3]])
  
y = np.array([1,1,-1,1,1,-1,-1,-1,1,1,1,1])
 
 
 
data = [[],[]]
result = []
 
x,y = genCircleInRect(data,result)
x = np.array(x)
y = np.array(y)
 
 
 
 
colormap = np.array(['placeholder','g','b']) #indexes considered are 1 and -1 hence 0 is of no use
 
weights = np.ones((1,len(x[0])))/len(x[0])
 
 
DS = DecisionStump()
DS.train(x,y,weights)
# print DS.error
classifiers = []
for i in xrange(100):
    DS = DecisionStump()
    DS.train(x,y,weights)
    classifiers.append(DS)
    error = DS.error
#     print DS.threshold
#     print DS.error
     
    if DS.feature == 0:
        ly = np.linspace(-10,10,100)
        lx = np.ones(100)*DS.threshold
    else:
        lx = np.linspace(-10,10,100)
        ly = np.ones(100)*DS.threshold
    plt.plot(lx,ly,"r^")
         
#     print DS.prediction
    alpha = float( 0.5 * log( (1.0-error) / max(error, 1e-16) ) )
    weights *= np.exp( -y*alpha*DS.prediction )
    weights /= weights.sum()
#     print weights
     
     
 
     
     
plt.scatter( x[0], x[1], c=colormap[y])
#plt.axis([0,6,0,6])
plt.show()
