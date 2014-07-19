import random
import numpy as np
from math import sqrt


class Layer:
    def __init__(self,nodeNumber,lastLayerNumber):
        self.nodes = np.zeros(nodeNumber)
                
        self.weights = np.random.uniform(-1.0,1.0,size=(nodeNumber,lastLayerNumber))
        