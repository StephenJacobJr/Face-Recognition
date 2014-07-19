
import numpy as np

def calcFeature1(intImg, x, y, width, height):
    "Computes 2 horizontal squares feature."
    middleX = x + width/2
    endX = x + width - 1
    endY = y + height - 1
    
    
    A = intImg[x][y]
    B = intImg[middleX][y]
    C = intImg[middleX][endY]
    D = intImg[x][endY]
    
    firstSquare = A + C - B - D
    
    E = B
    F = intImg[endX][y]
    G = intImg[endX][endY]
    H = C
    
    secondSquare = E + G - F - H
    
    return secondSquare - firstSquare

def calcFeature2(intImg, x, y, width, height):
    "Computes 2 vertical squares feature."
    middleY = y + height/2
    endX = x + width - 1
    endY = y + height - 1
    
    A = intImg[x][y]
    B = intImg[endX][y]
    C = intImg[endX][middleY]
    D = intImg[x][middleY]
    
    firstSquare = A + C - B - D
    
    E = D
    F = C
    G = intImg[endX][endY]
    H = intImg[x][endY]
    
    secondSquare = E + G - F - H  
      
    return secondSquare - firstSquare

def calcFeature3(intImg, x, y, width, height):
    "Computes 3 horizontal squares feature."
    middleX1 = x + width/3
    middleX2 = x + 2*width/3
    endX = x + width - 1
    endY = y + height - 1
    
    
    A = intImg[x][y]
    B = intImg[middleX1][y]
    C = intImg[middleX1][endY]
    D = intImg[x][endY]
    
    firstSquare = A + C - B - D
    
    E = B
    F = intImg[middleX2][y]
    G = intImg[middleX2][endY]
    H = C
    
    secondSquare = E + G - F - H
    
    I = F
    J = intImg[endX][y]
    K = intImg[endX][endY]
    L = G
    
    thirdSquare = I + K - J - L
    
    return secondSquare - (firstSquare + thirdSquare)

def calcFeature4(intImg, x, y, width, height):
    "Computes 3 vertical squares feature."
    middleY1 = y + height/3
    middleY2 = y + 2*height/3
    
    endX = x + width - 1
    endY = y + height - 1
    
    A = intImg[x][y]
    B = intImg[endX][y]
    C = intImg[endX][middleY1]
    D = intImg[x][middleY1]
    
    firstSquare = A + C - B - D
    
    E = D
    F = C
    G = intImg[endX][middleY2]
    H = intImg[x][middleY2]
    
    secondSquare = E + G - F - H
    
    I = H 
    J = G
    K = intImg[endX][endY]
    L = intImg[x][endY]
    
    thirdSquare = I + K - J - L
    
    return secondSquare - (firstSquare + thirdSquare)

def calcFeature5(intImg, x, y, width, height):
    "Computes 4 diagonal squares feature."
    middleX = x + width/2
    middleY = y + height/2
    endX = x + width - 1
    endY = y + height - 1
    
    A = intImg[x][y]
    B = intImg[middleX][y]
    C = intImg[middleX][middleY]
    D = intImg[x][middleY]
    
    firstSquare = A + C - B - D
    
    E = B
    F = intImg[endX][y]
    G = intImg[endX][middleY]
    H = C
    
    secondSquare = E + G - F - H
    
    I = D
    J = C
    K = intImg[middleX][endY]
    L = intImg[x][endY]
    
    thirdSquare = I + K - J - L
    
    M = C
    N = G
    O = intImg[endX][endY]
    P = K
    
    fourthSquare = M + O - N - P
    
    return (secondSquare + thirdSquare) - (firstSquare + fourthSquare)


    

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


def calcSingleFeature(indexTable,index,integralImage):    
    featuresCalc = [calcFeature1,calcFeature2,calcFeature3,calcFeature4,calcFeature5]
    featIndex = indexTable[index][0]
    x,y = indexTable[index][1],indexTable[index][2]
    curWidth,curHeight = indexTable[index][3],indexTable[index][4]
    
    return np.array([featuresCalc[featIndex](integralImage, x, y, curWidth, curHeight)])