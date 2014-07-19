

import Image
import numpy as np
from ImageProcessor import *

img = Image.open("face.png")
 
#grayscale (L for luminance)
 
img = img.convert('L')
img = np.array(img,dtype='int64') # to avoid overflow
print "Shape:",img.shape
getFeatures(img)