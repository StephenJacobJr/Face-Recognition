# 
# import numpy as np
# import Image
# from Network import Network
# import matplotlib.pyplot as plt
# from matplotlib.mlab import find
# img = Image.open("test.png")
# 
# #grayscale (L for luminance)
# 
# img = img.convert('L')
# 
# # print np.array(img)
# # print np.array(img).shape
# 
# 
# 
# def getIntegralImage(imgArray):
#     return imgArray.cumsum(0).cumsum(1)
#     
# # print getIntegralImage(np.array(img))
# 
# Network((2,1))
# 
# origx = x
# x = np.array([[1,1,1,1,1,1,1,1],
#               [2,2,3,4,4,3.1,2.9,2.9],
#               [2,4,3,2,4,3.1,2.9,3.0]])
# 
# y = np.array([1,1,-1,1,1,-1,-1,-1])
# 
# weights = [[1.0/5,1.0/5,1.0/5,1.0/5,1.0/5,1.0/5,1.0/5,1.0/5]]
# 
# 
# indexes = range(len(x[0]))
# 
# dataset = [(2,2,1),(2,4,1),(3,3,-1),(4,2,1),(4,4,1)]
# 
# N = Network((3,1))
# N.layers[1].weights = np.array([[1.0/5,1.0/5,1.0/5]])
# 
# for k in xrange(17):    
#     print k
#     N.layers[1].weights = np.array([[1.0/5,1.0/5,1.0/5]])
#     for j in xrange(10):
#         for i in indexes:
#             N.setInput( (x[0][i], x[1][i], x[2][i]) )
#             N.update()
#             N.backprop( y[i] )
#     
#     
#     w = N.layers[1].weights[0]
#     a = -w[1]/w[2]
#     b = -w[0]/w[2]
#     nnx = np.linspace(0,6,100)
#     nny = a*nnx +b
#     
#     plt.plot(nnx,nny) 
#     
#     
#     
#     
# 
# for i in xrange(len(x[0])):
#     N.setInput( (x[0][i], x[1][i], x[2][i]) )
#     print  (x[1][i],x[2][i]), ":", y[i]
#     print N.update()
#     
# 
#         
# 
# 
# w = N.layers[1].weights[0]
# a = -w[1]/w[2]
# b = -w[0]/w[2]
# nnx = np.linspace(0,6,100)
# nny = a*nnx +b
# 
# plt.plot(nnx,nny) 
# plt.plot( x[1], x[2], "ro")
# plt.axis([0,6,0,6])
# plt.show()




