
import numpy as np
import time

def calcFeature1v(intImg, X, Y, MIDDLEX, ENDX, ENDY):
    
    
    
    A = intImg[X,Y]
    B = intImg[MIDDLEX,Y]
    C = intImg[MIDDLEX,ENDY]
    D = intImg[X,ENDY]
    
    firstSquare = A + C - B - D
    
    E = B
    F = intImg[ENDX,Y]
    G = intImg[ENDX,ENDY]
    H = C
    
    secondSquare = E + G - F - H
    
    result = secondSquare - firstSquare
    
    
    return result




def calcFeature2v(intImg, X, Y, MIDDLEY, ENDX, ENDY):
    "Computes 2 vertical squares feature."    
    
    A = intImg[X,Y]
    B = intImg[ENDX,Y]
    C = intImg[ENDX,MIDDLEY]
    D = intImg[X,MIDDLEY]
    
    firstSquare = A + C - B - D
    
    E = D
    F = C
    G = intImg[ENDX,ENDY]
    H = intImg[X,ENDY]
    
    secondSquare = E + G - F - H  
      
    result = secondSquare - firstSquare
    return result

def calcFeature3v(intImg, X, Y, THIRDX1, THIRDX2, ENDX, ENDY):
    "Computes 3 horizontal squares feature."
    
    
    A = intImg[X,Y]
    B = intImg[THIRDX1,Y]
    C = intImg[THIRDX1,ENDY]
    D = intImg[X,ENDY]
    
    firstSquare = A + C - B - D
    
    E = B
    F = intImg[THIRDX2,Y]
    G = intImg[THIRDX2,ENDY]
    H = C
    
    secondSquare = E + G - F - H
    
    I = F
    J = intImg[ENDX,Y]
    K = intImg[ENDX,ENDY]
    L = G
    
    thirdSquare = I + K - J - L
    
    result = secondSquare - (firstSquare + thirdSquare)
    return result



def calcFeature4v(intImg, X, Y, THIRDY1, THIRDY2, ENDX, ENDY):
    "Computes 3 vertical squares feature."
    
    A = intImg[X,Y]
    B = intImg[ENDX,Y]
    C = intImg[ENDX,THIRDY1]
    D = intImg[X,THIRDY1]
    
    firstSquare = A + C - B - D
    
    E = D
    F = C
    G = intImg[ENDX,THIRDY2]
    H = intImg[X,THIRDY2]
    
    secondSquare = E + G - F - H
    
    I = H 
    J = G
    K = intImg[ENDX,ENDY]
    L = intImg[X,ENDY]
    
    thirdSquare = I + K - J - L
    
    result = secondSquare - (firstSquare + thirdSquare)
    return result




def calcFeature5v(intImg, X, Y, MIDDLEX, MIDDLEY, ENDX, ENDY):
    "Computes 4 diagonal squares feature."
    
    A = intImg[X,Y]
    B = intImg[MIDDLEX,Y]
    C = intImg[MIDDLEX,MIDDLEY]
    D = intImg[X,MIDDLEY]
    
    firstSquare = A + C - B - D
    
    E = B
    F = intImg[ENDX,Y]
    G = intImg[ENDX,MIDDLEY]
    H = C
    
    secondSquare = E + G - F - H
    
    I = D
    J = C
    K = intImg[MIDDLEX,ENDY]
    L = intImg[X,ENDY]
    
    thirdSquare = I + K - J - L
    
    M = C
    N = G
    O = intImg[ENDX,ENDY]
    P = K
    
    fourthSquare = M + O - N - P
    
    result = secondSquare + thirdSquare - firstSquare - fourthSquare
    return result
    

def getIntegralImage(imgArray):
    return imgArray.cumsum(0).cumsum(1)
    
def genIndexTable(shape):
    results = []
    features = [[2,1],[1,2],[3,1],[1,3],[2,2]]
    sizeX,sizeY = shape
    for featIndex in xrange(len(features)):
        featSizeX = features[featIndex][0]
        featSizeY = features[featIndex][1]        
        for width in xrange(sizeX/featSizeX):
            curWidth = (width+1)*featSizeX
            for height in xrange(sizeY/featSizeY):
                curHeight = (height+1)*featSizeY
                for x in xrange(sizeX - curWidth + 1):
                    for y in xrange(sizeY - curHeight + 1):
                        result = [featIndex, x, y, curWidth, curHeight]
                        results.append(result)
    return np.array(results)



def calcImgData(classifierFeatures,imgArray, indexTable):
    cF = classifierFeatures
    
    integralImage = getIntegralImage(imgArray)
    featTypes = indexTable[cF,0]
    X = indexTable[cF,1]
    Y = indexTable[cF,2]
    WIDTH = indexTable[cF,3]
    HEIGHT = indexTable[cF,4]  
    
    hw = WIDTH/2
    tw = WIDTH/3
    hh = HEIGHT/2
    th = HEIGHT/3
      
    MIDDLEX = X + hw
    MIDDLEY = Y + hh
    THIRDX1 = X + tw
    THIRDX2 = X + + tw + tw
    THIRDY1 = Y + th
    THIRDY2 = Y + th + th
    ENDX = X + WIDTH - 1
    ENDY = Y + HEIGHT - 1
    
    
    Xs = []
    Ys=[]
    ENDXs=[]
    ENDYs=[]
    MIDDLEXs=[]
    MIDDLEYs=[]
    THIRDX1s=[]
    THIRDX2s=[]
    THIRDY1s=[]
    THIRDY2s=[]
        
    rightFeats = []
    for i in xrange(5):    
        rightFeat = featTypes == i
        rightFeats.append(rightFeat)
        Xs.append(X[rightFeat])
        Ys.append(Y[rightFeat])
        ENDXs.append(ENDX[rightFeat])
        ENDYs.append(ENDY[rightFeat])
        MIDDLEXs.append(MIDDLEX[rightFeat])
        MIDDLEYs.append(MIDDLEY[rightFeat])
        THIRDX1s.append(THIRDX1[rightFeat])
        THIRDX2s.append(THIRDX2[rightFeat])
        THIRDY1s.append(THIRDY1[rightFeat])
        THIRDY2s.append(THIRDY2[rightFeat])
        
        
    return [integralImage, rightFeats, Xs, Ys, ENDXs, ENDYs, MIDDLEXs, MIDDLEYs, THIRDX1s, THIRDX2s, THIRDY1s, THIRDY2s]

    
def genFeatureSet(featShape, integralImage, featSortedIndexes, X, Y, ENDX, ENDY, MIDDLEX, MIDDLEY, THIRDX1, THIRDX2, THIRDY1, THIRDY2):    
    
    
    #indexes are which features are selected
    #featType is type (0-5) of each feature
    
    #to speed up: create new function to vectorize whole image calculations and not sliding window 90 000 range loop
    #keep this one for cascading
             
    features = np.empty(featShape)
    
   
    f1 = calcFeature1v(integralImage,  X[0], Y[0], MIDDLEX[0], ENDX[0], ENDY[0])
    f2 = calcFeature2v(integralImage,  X[1], Y[1], MIDDLEY[1], ENDX[1], ENDY[1])
    f3 = calcFeature3v(integralImage,  X[2], Y[2], THIRDX1[2], THIRDX2[2], ENDX[2], ENDY[2])
    f4 = calcFeature4v(integralImage,  X[3], Y[3], THIRDY1[3], THIRDY2[3], ENDX[3], ENDY[3])
    f5 = calcFeature5v(integralImage,  X[4], Y[4], MIDDLEX[4], MIDDLEY[4], ENDX[4], ENDY[4])
   
    fs = [f1,f2,f3,f4,f5]
    
    for i in xrange(5):
        features[featSortedIndexes[i]] = fs[i] #this step so that the feature remain in the same order as the sides and alphas and thresholds


    
    return features

    
    
    
    
    
    
