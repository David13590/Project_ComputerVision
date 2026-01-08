import cv2  
import numpy as np 
from matplotlib import pyplot as plot 

imgResizeX = 500
imgResizeY = 400

img_original = cv2.imread("images/hylstre_maatte.jpg")
img_grayscale = cv2.imread("images/hylstre_maatte.jpg", 0) #Convert to grayscale

def img_plot():
    plot.subplot(121)
    cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale", imgResizeX, imgResizeY)
    cv2.imshow("Grayscale", img_grayscale)

    plot.subplot(122)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", imgResizeX, imgResizeY)
    cv2.imshow("Original", img_original)

    plot.tight_layout()
    plot.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_histograms():
    grayscale_histogram = cv2.calcHist([img_grayscale], [0], None, [256], [0,256]) # Create histogram of grayscale img

    # Create equalized grayscale image and histogram
    grayscale_BGR2GRAY = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(grayscale_BGR2GRAY)
    equalized_histogram = cv2.calcHist([equalized_img], [0], None, [256], [0,256])

    # Plot images and histograms
    plot.subplot(221)
    plot.imshow(img_grayscale, cmap='gray')
    plot.title('Original Grayscale')
    plot.axis('off')

    plot.subplot(222)
    plot.plot(grayscale_histogram, color='k')
    plot.title('Histogram Original')
    plot.xlim([0,256])

    plot.subplot(223)
    plot.imshow(equalized_img, cmap='gray')
    plot.title('Equalized Grayscale')
    plot.axis('off')

    plot.subplot(224)
    plot.plot(equalized_histogram, color='k')
    plot.title('Histogram Equalized')
    plot.xlim([0,256])

    plot.tight_layout()
    plot.show()

img_plot()
compare_histograms()
