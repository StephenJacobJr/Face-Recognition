
import numpy as np
import time

def calcFeature1v(intImg, featTypes, X, Y, MIDDLEX, ENDX, ENDY):
    
    X = X[featTypes == 0]
    Y = Y[featTypes == 0]
    MIDDLEX = MIDDLEX[featTypes == 0]
    ENDX = ENDX[featTypes == 0]
    ENDY = ENDY[featTypes == 0]
    
    
    
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




def calcFeature2v(intImg, featTypes, X, Y, MIDDLEY, ENDX, ENDY):
    "Computes 2 vertical squares feature."    
    
    
    X = X[featTypes == 1]
    Y = Y[featTypes == 1]
    MIDDLEY = MIDDLEY[featTypes == 1]
    ENDX = ENDX[featTypes == 1]
    ENDY = ENDY[featTypes == 1]
    
    
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

def calcFeature3v(intImg, featTypes, X, Y, THIRDX1, THIRDX2, ENDX, ENDY):
    "Computes 3 horizontal squares feature."
    
    X = X[featTypes == 2]
    Y = Y[featTypes == 2]
    THIRDX1 = THIRDX1[featTypes == 2]
    THIRDX2 = THIRDX2[featTypes == 2]
    ENDX = ENDX[featTypes == 2]
    ENDY = ENDY[featTypes == 2]
    
    
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



def calcFeature4v(intImg, featTypes, X, Y, THIRDY1, THIRDY2, ENDX, ENDY):
    "Computes 3 vertical squares feature."
    
    
    X = X[featTypes == 3]
    Y = Y[featTypes == 3]
    THIRDY1 = THIRDY1[featTypes == 3]
    THIRDY2 = THIRDY2[featTypes == 3]
    ENDX = ENDX[featTypes == 3]
    ENDY = ENDY[featTypes == 3]
    
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




def calcFeature5v(intImg, featTypes, X, Y, MIDDLEX, MIDDLEY, ENDX, ENDY):
    "Computes 4 diagonal squares feature."
    
    X = X[featTypes == 4]
    Y = Y[featTypes == 4]
    MIDDLEX = MIDDLEX[featTypes == 4]
    MIDDLEY = MIDDLEY[featTypes == 4]
    ENDX = ENDX[featTypes == 4]
    ENDY = ENDY[featTypes == 4]
    
    
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
    
    result = (secondSquare + thirdSquare) - (firstSquare + fourthSquare)
    return result
    

def getIntegralImage(imgArray):
    return imgArray.cumsum(0).cumsum(1)
    

def getFeatures(imgArray):
    integralImage = getIntegralImage(imgArray)
    results = []
    features = [[2,1],[1,2],[3,1],[1,3],[2,2]]
    featuresCalc = [calcFeature1,calcFeature2,calcFeature3,calcFeature4,calcFeature5]
    sizeX,sizeY = imgArray.shape
    for featIndex in xrange(len(features)):
        featSizeX = features[featIndex][0]
        featSizeY = features[featIndex][1]
        
        
        for width in xrange(sizeX/featSizeX):            
            curWidth = (width+1)*featSizeX
            
            for height in xrange(sizeY/featSizeY):
                curHeight = (height+1)*featSizeY
                
                for x in xrange(sizeX - curWidth + 1):
                    for y in xrange(sizeY - curHeight + 1):
                        
                        result = featuresCalc[featIndex](integralImage, x, y, curWidth, curHeight)
#                         result = 1
                        results.append([result])
#     print len(results)
    return np.array(results)

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

    
def genFeatureSet2(indexes, featTypes, X, Y, WIDTH, HEIGHT, integralImage):    
    
    
    #indexes are which features are selected
    #featType is type (0-5) of each feature
    
          
          
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
    
    features = np.empty(indexes.shape)
    
    f1 = calcFeature1v(integralImage, featTypes, X, Y, MIDDLEX, ENDX, ENDY)
    f2 = calcFeature2v(integralImage, featTypes, X, Y, MIDDLEY, ENDX, ENDY)
    f3 = calcFeature3v(integralImage, featTypes, X, Y, THIRDX1, THIRDX2, ENDX, ENDY)
    f4 = calcFeature4v(integralImage, featTypes, X, Y, THIRDY1, THIRDY2, ENDX, ENDY)
    f5 = calcFeature5v(integralImage, featTypes, X, Y, MIDDLEX, MIDDLEY, ENDX, ENDY)
    
    fs = [f1,f2,f3,f4,f5]
    
    for i in xrange(5):
        features[featTypes==i] = fs[i] #this step so that the feature remain in the same order as the sides and alphas and thresholds
    
    
    return features


