import cv2  
import numpy as np 
from matplotlib import pyplot as plot 

imgResizeX = 500
imgResizeY = 400

grayscale = cv2.imread("images/hylstre_maatte.jpg", 0) #Convert to grayscale

cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Grayscale", imgResizeX, imgResizeY)
cv2.imshow("Grayscale", grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()

grayscale_histogram = cv2.calcHist([grayscale],[0],None,[256],[0,256]) #Create histogram of grayscale img

plot.subplot(221), plot.imshow(grayscale, cmap="Grays")
plot.subplot(222), plot.plot(grayscale_histogram)
plot.xlim(0, 256)

plot.show()
