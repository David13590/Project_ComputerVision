import cv2  
import numpy as np 
from matplotlib import pyplot as plot 

img_original_casings_black = cv2.imread("images/hylstre_maatte.jpg")
img_grayscale_casings_black = cv2.imread("images/hylstre_maatte.jpg", 0) #Convert to grayscale

img_original_casings_noise = cv2.imread("images/hylstre_stoej.jpg")
img_grayscale_casings_noise = cv2.imread("images/hylstre_stoej.jpg", 0) #Convert to grayscale

def img_plot(input_img_original, input_img_grayscale):
    plot.figure(1) # Create and activete figure 1

    plot.subplot(121)
    plot.imshow(input_img_original)
    plot.title('Original Image')
    plot.axis('off')

    plot.subplot(122)
    plot.imshow(input_img_grayscale, cmap='gray')
    plot.title('Grayscale Image')
    plot.axis('off')

    plot.tight_layout()

def compare_histograms(input_img_grayscale, input_img_original):
    grayscale_histogram = cv2.calcHist([input_img_grayscale], [0], None, [256], [0,256]) # Create histogram of grayscale img

    # Create equalized grayscale image and histogram
    grayscale_BGR2GRAY = cv2.cvtColor(input_img_original, cv2.COLOR_BGR2GRAY)
    img_equalized = cv2.equalizeHist(grayscale_BGR2GRAY)
    equalized_histogram = cv2.calcHist([img_equalized], [0], None, [256], [0,256])

    # Plot images and histograms
    plot.figure(2) # Create and activate figure 2

    plot.subplot(221)
    plot.imshow(input_img_grayscale, cmap='gray')
    plot.title('Original Grayscale')
    plot.axis('off')

    plot.subplot(222)
    plot.plot(grayscale_histogram, color='k')
    plot.title('Histogram Original')
    plot.xlim([0,256])

    plot.subplot(223)
    plot.imshow(img_equalized, cmap='gray')
    plot.title('Equalized Grayscale')
    plot.axis('off')

    plot.subplot(224)
    plot.plot(equalized_histogram, color='k')
    plot.title('Histogram Equalized')
    plot.xlim([0,256])

    plot.tight_layout()

    return input_img_grayscale, img_equalized

# Contour detect 
def contour_detect(img, img2):
    thresh_lower_bound = 200
    thresh_upper_bound = 255
    thresh_type = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

    contour_line_thickness = 2
    contour_parrent_idx = -1 # -1 to draw all contours
    contour_line_color = (0, 255, 0) # BGR
    contour_min_area = 200  # Minimum area to consider a contour

    # First img
    ret, thresh = cv2.threshold(img, thresh_lower_bound, thresh_upper_bound, thresh_type[0]) # Apply threshold for better accuracy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in thresholded image
    contour_filtered = [c for c in contours if cv2.contourArea(c) >= contour_min_area]  # Filter contours by area

    # Draw contours on a copy of the original image
    img_contours = img.copy()
    cv2.drawContours(img_contours, contour_filtered, contour_parrent_idx, contour_line_color, contour_line_thickness) # Draw all contours
    
    # Second img
    ret2, thresh2 = cv2.threshold(img2, thresh_lower_bound, thresh_upper_bound, thresh_type[0]) # Apply threshold for better accuracy
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in thresholded image 
    contour2_filtered = [c for c in contours2 if cv2.contourArea(c) >= contour_min_area]  # Filter contours by area

    img2_contours = img2.copy()
    cv2.drawContours(img2_contours, contour2_filtered, contour_parrent_idx, contour_line_color, contour_line_thickness) # Draw all contours

    # Plot contours
    plot.figure(3) # Create and activate figure 3

    plot.subplot(211)
    plot.imshow(img_contours, cmap='gray')
    plot.text(0.00, 0.95, f"count: {len(contour_filtered)}", transform=plot.gca().transAxes, fontsize=9, color='red') # Add contour count text
    plot.title('Contours')
    plot.axis('off')

    plot.subplot(212)
    plot.imshow(img2_contours, cmap='gray')
    plot.text(0.00, 0.95, f"count: {len(contour2_filtered)}", transform=plot.gca().transAxes, fontsize=9, color='red') # Add contour count text
    plot.title('Contours Equalized')
    plot.axis('off')
    
    return contours, img_contours, contours2, img2_contours

# Call plot functions
# First set of images
img_plot(img_original_casings_black, img_grayscale_casings_black) 
img_grayscale_hist, img_equalized_hist = compare_histograms(img_grayscale_casings_black, img_original_casings_black) # Save returned values
contours_grayscale = contour_detect(img_grayscale_hist, img_equalized_hist)

plot.show() # Show all plots
cv2.waitKey(0)
cv2.destroyAllWindows()

# Second set of images
img_plot(img_original_casings_noise, img_grayscale_casings_noise)
img2_grayscale_hist, img2_equalized_hist = compare_histograms(img_grayscale_casings_noise, img_original_casings_noise) # Save returned values
contours2_grayscale = contour_detect(img2_grayscale_hist, img2_equalized_hist)

plot.show() # Show all plots
cv2.waitKey(0)
cv2.destroyAllWindows()
