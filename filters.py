import cv2  
import numpy as np 
from matplotlib import pyplot as plot 

img_original_casings_black = cv2.imread("images/hylstre_maatte.jpg")
img_grayscale_casings_black = cv2.imread("images/hylstre_maatte.jpg", 0) #Convert to grayscale

img_original_casings_noise = cv2.imread("images/hylstre_stoej.jpg")
img_grayscale_casings_noise = cv2.imread("images/hylstre_stoej.jpg", 0) #Convert to grayscale

def img_plot(input_img_original, input_img_grayscale):
    plot.figure(1) # Create and activate figure 1

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
    thresh_lower_bound = 180
    thresh_upper_bound = 255
    thresh_type = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

    contour_line_thickness = 2
    contour_parrent_idx = -1 # -1 to draw all contours
    contour_line_color = (0, 255, 0) # BGR
    contour_min_area = 800  # Minimum area in pixels to consider as contour
    contour_max_area = 20000

    plot_font_size = 9
    plot_text_color = 'red'
    plot_text_position_x = 0.00 
    plot_text_position_y = 0.95

    # First img
    ret, thresh = cv2.threshold(img, thresh_lower_bound, thresh_upper_bound, thresh_type[0]) # Apply threshold (min, max intensity)
    plot.figure(6)
    plot.subplot(121)
    plot.imshow(thresh)
    plot.title('Thresholded Image fig6')
    plot.axis('off')
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours
    contour_filtered = [c for c in contours if cv2.contourArea(c) >= contour_min_area]  # Filter contours by area (min, max area)
    contour_filtered2 = [c for c in contour_filtered if cv2.contourArea(c) <= contour_max_area] 
    
    # Draw contours on first img
    img_contours = img.copy()
    cv2.drawContours(img_contours, contour_filtered2, contour_parrent_idx, contour_line_color, contour_line_thickness) # Draw all contours
    plot.figure(6)
    plot.subplot(122)
    plot.imshow(img_contours)
    plot.title('Thresholded contour filtered Image')
    plot.axis('off')
    
    # Second img
    ret2, thresh2 = cv2.threshold(img2, thresh_lower_bound, thresh_upper_bound, thresh_type[0]) # Apply threshold
    plot.figure(8)
    plot.subplot(121)
    plot.imshow(thresh2)
    plot.title('Thresholded Image fig8')
    plot.axis('off')
    
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours 
    contour2_filtered = [c for c in contours2 if cv2.contourArea(c) >= contour_min_area]  # Filter contours by area
    contour2_filtered2 = [c for c in contour2_filtered if cv2.contourArea(c) <= contour_max_area] 

    img2_contours = img2.copy()
    cv2.drawContours(img2_contours, contour2_filtered2, contour_parrent_idx, contour_line_color, contour_line_thickness) # Draw all contours
    plot.figure(8)
    plot.subplot(122)
    plot.imshow(img2_contours)
    plot.title('Thresholded contour filtered Image')
    plot.axis('off')

    # Plot contours
    plot.figure(3) # Create and activate figure 3

    plot.subplot(211)
    plot.imshow(img_contours, cmap='gray')
    plot.text(plot_text_position_x, plot_text_position_y, f"count: {len(contour_filtered2)}", transform=plot.gca().transAxes, fontsize=plot_font_size, color=plot_text_color) # Add contour count text
    plot.title('Contours')
    plot.axis('off')

    plot.subplot(212)
    plot.imshow(img2_contours, cmap='gray')
    plot.text(plot_text_position_x, plot_text_position_y, f"count: {len(contour2_filtered2)}", transform=plot.gca().transAxes, fontsize=plot_font_size, color=plot_text_color) # Add contour count text
    plot.title('Contours Equalized')
    plot.axis('off')
    
    return contours, img_contours, contours2, img2_contours

def otsu_binarization(input_img_equalized_grayscale, input_img2_original_grayscale):
    otsu_thresh_lower_bound = 0
    otsu_thresh_upper_bound = 255
    gaussian_kernel_size = (25,25)

    # Original image
    # Set 1
    histogram_original = cv2.calcHist([input_img2_original_grayscale], [0], None, [256], [0,256])
    ret1, otsu_thresh_original = cv2.threshold(input_img2_original_grayscale, otsu_thresh_lower_bound, otsu_thresh_upper_bound, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu binarization
    histogram_otsu_original = cv2.calcHist([otsu_thresh_original], [0], None, [256], [0,256])

    # Set 2
    gaussian_blur = cv2.GaussianBlur(input_img2_original_grayscale, gaussian_kernel_size, 0) # Gaussian blur to reduce noise
    histogram_gaussian = cv2.calcHist([gaussian_blur], [0], None, [256], [0,256]) # calc histogram
    ret1, otsu_thresh_gaussian = cv2.threshold(gaussian_blur, otsu_thresh_lower_bound, otsu_thresh_upper_bound, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu binarization
    
    
    histogram_otsu_gaussian = cv2.calcHist([otsu_thresh_gaussian], [0], None, [256], [0,256])

    # Plot images and histograms
    plot.figure(4)

    # Set 1
    plot.subplot(231)
    plot.imshow(input_img2_original_grayscale, cmap='gray')
    plot.title('Original Grayscale')
    plot.axis('off')

    plot.subplot(232)
    plot.plot(histogram_original, color='k')
    plot.title('Histogram Original')
    plot.xlim([0,236])

    plot.subplot(233)
    plot.imshow(otsu_thresh_original)
    plot.title('Otsu threshold')
    plot.axis('off')

    

    # Set 2
    plot.subplot(234)
    plot.imshow(gaussian_blur, cmap='gray')
    plot.title('Gaussian Blurred original')
    plot.axis('off')

    plot.subplot(235)
    plot.plot(histogram_gaussian, color='k')
    plot.title('Histogram Gaussian Blurred')
    plot.xlim([0,256])

    plot.subplot(236)
    plot.imshow(otsu_thresh_gaussian)
    plot.title('Otsu threshold')
    plot.axis('off')

    plot.tight_layout()
    
    return otsu_thresh_gaussian, otsu_thresh_original


# # Call plot functions
# # First set of images
# img_plot(img_original_casings_black, img_grayscale_casings_black) 
# img_grayscale_hist, img_equalized_hist = compare_histograms(img_grayscale_casings_black, img_original_casings_black) # Save returned values
# contours_grayscale = contour_detect(img_grayscale_hist, img_equalized_hist)

# plot.show() # Show all plots
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Second set of images
#img_plot(img_original_casings_noise, img_grayscale_casings_noise)
img2_grayscale_hist, img2_equalized_hist = compare_histograms(img_grayscale_casings_noise, img_original_casings_noise) # Save returned values
otsu_binarization_img_gaussian, otsu_binarization_img_original = otsu_binarization(img2_equalized_hist, img2_grayscale_hist) # Do otsu binerization before contour detection
#contour_detect(img2_grayscale_hist, img2_equalized_hist)
#contour_detect(otsu_binarization_img_original, otsu_binarization_img_gaussian)

opening_original = cv2.morphologyEx(otsu_binarization_img_original, cv2.MORPH_OPEN, np.ones((32,32), np.uint8))
opening_equalized = cv2.morphologyEx(otsu_binarization_img_gaussian, cv2.MORPH_OPEN, np.ones((32,32), np.uint8))
img_plot(opening_original, opening_equalized)
contour_detect(opening_original, opening_equalized)

plot.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
