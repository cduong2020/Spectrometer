import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy import signal


from scipy.optimize import curve_fit
np.set_printoptions(threshold=sys.maxsize)
from skimage.io import imread, imshow
from math import sqrt,exp,floor


#############################################################
#
# Global Variables
#
#############################################################
laserOne = cv2.cvtColor(cv2.imread("images/405nm_laser_2.jpg"), cv2.COLOR_BGR2GRAY)
laserTwo = cv2.cvtColor(cv2.imread("images/532nm_laser_2.jpg"), cv2.COLOR_BGR2GRAY)
laserThree =   cv2.cvtColor(cv2.imread("images/650nm_laser_2.jpg"), cv2.COLOR_BGR2GRAY)
laserWavelengths = [405, 532, 650]


FWHM = []   # Will hold a FWHM for each laser
centerY     # Calculated average center row to be analyzed
linearEq    # [m, b]
cubicEq     # [x3, x2, x, c]


#############################################################
#
# Helper Functions
#
#############################################################

def spectrumCenter(image):  # Called for each laser line
    # Returns [x,y] of spectrum center
    edges = cv2.Canny(image, threshold1=30, threshold2=255) #(image, minValue, maxValue)
    pixels = np.where(edges >= 200)  # Array of coordinate arrays [ [y values], [x values] ]
    min_X = np.min(pixels[1])
    max_X = np.max(pixels[1])
    avg_X = (min_X + max_X)/2 
    min_Y = np.min(pixels[0])
    max_Y = np.max(pixels[0])
    avg_Y = (min_Y + max_Y)/2 
    print(avg_X, avg_Y)
    plt.imshow(edges, cmap="gray")
    plt.show()
    return avg_Y, avg_X

def gauss(x, H, A, x0, sigma):
    # Returns y values for a gaussian 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    return popt

# Operates on image to fit a gaussian at a given row, Returns FWHM, x_center
def gaussianFitImage(image, row):
    # 2D Image Array
    row = round(row)  # round the row to closest integer pixel 
    filtered = cv2.fastNlMeansDenoising(image)
    intensity = np.sum(filtered[(row-5):(row+5), 0:image.shape[1]], 0)/10
    peak = np.where(intensity == np.max(intensity))[0][0]
    xdata = np.linspace((peak-15), (peak+15), 30)
    ydata = intensity[(peak-15):(peak+15)]
    H, A, x0, sigma = gauss_fit(xdata, ydata)   #
    FWHM = 2.355 * sigma # The expected relationship between std and FWHM is 2.355  https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    return FWHM, x0


# 2d PSF
def GaussianPSF(intensity, radius, imgShape):
    #Intensity Range 0-1
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(radius**2))))
    return intensity*base

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def closest(lst, K):
      
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return idx



#############################################################
#
# Spectrum Capture Function
# - Returns nothing
# - images are stored in '.images/captures'
# - merged final image is stored in ',images/exposureFusion.bmp'
#
#############################################################

# https://docs.opencv.org/master/d3/db7/tutorial_hdr_imaging.html
# https://learnopencv.com/exposure-fusion-using-opencv-cpp-python/
# https://www.scitepress.org/Papers/2014/50872/50872.pdf
def captureSpectrum():
    cam = cv2.VideoCapture(0)       # Selects Camera Input 0, 1, 2...
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    img_exp = -10
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25) # turn off auto exposure
    cam.set(cv2.CAP_PROP_EXPOSURE, img_exp)     # Exposure values follow power or 2's Ex: -1 => 2^-1 => 1/2s
    
    # manual gain and brightness may not be needed
    # if it is we will need a way to determine values
    # cam.set(cv2.CAP_PROP_GAIN, 1000)
    # cam.set(cv2.CAP_PROP_BRIGHTNESS, 0)

    img_counter = 0
    pics = []
    for img_counter in range(11):
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        pics.append(frame)
        img_name = "frame_{}.bmp".format(img_counter)
        cv2.imwrite('./images/captures/'+img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        img_exp += 0.5
        cam.set(cv2.CAP_PROP_EXPOSURE, img_exp)
    

    # Align input images
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(pics, pics)
    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(pics)
    cv2.imwrite("images/exposureFusion.bmp",exposureFusion*255)
    
    img= cv2.imread("images/exposureFusion.bmp")
    dst = cv2.fastNlMeansDenoisingColored(img,None,5,5,7,21) #h(for luminance component)=5 hColor (for color component)=5 tempWindowSize= 7 (default) searchWindowSize = 21 (default) 
    img_name = "images/denoise_frame.bmp"
    cv2.imwrite(img_name, dst)
    
    cam.release()
    cv2.destroyAllWindows()
    return


#############################################################
#
# Laser Calbration Function
# - Returns avg_Y, FWHM: [.., .. ,..], linearEq: [m, b]
#
#############################################################

def linearCalibration(laserImgs, wavelengths): # Expects ([img1, img2, img3], [num1, num2, num3])
    # Find the center of each laser line
    center_y1, center_x1 =  spectrumCenter(laserImgs[0])
    center_y2, center_x2 =  spectrumCenter(laserImgs[1])
    center_y3, center_x3 =  spectrumCenter(laserImgs[2])

    # Average Y center is the row used for whole spectrum analysis
    avg_Y = round((center_y1 + center_y2 + center_y3)/3)

    # Find average FWHM and X Coord for each laser line
    FWHM_1, x_1 = gaussianFitImage(laserImgs[0], center_y1)
    FWHM_2, x_2 = gaussianFitImage(laserImgs[1], center_y2)
    FWHM_3, x_3 = gaussianFitImage(laserImgs[2], center_y3)

    # Group all FWHM values    
    FWHM = ([FWHM_1, FWHM_2, FWHM_3])

    # Fit Linear Function
    x = [x_1, x_2, x_3]
    y = [wavelengths[0], wavelengths[1], wavelengths[2]]
    linearEq = np.polyfit(x, y, 1)
    return avg_Y, FWHM, linearEq


#############################################################
#
# Analyze Spectrum 
# - Used for all spectrum including fluorescent calibration
# - Returns x_coordinates[], wavelengths[], peaks[]
#
#############################################################

def analyzeSpectrum(image, row, isLinear):
    row = round(row)
    intensity = np.sum(image[(row-15):(row+15), 0:image.shape[1]], 0)/30     # Create 1D Array using average intensity +-30 from center

    sigma = ((FWHM[0] + FWHM[1] + FWHM[2])/3)/2.355                             # Average Sigma Found While Calibrating

    # Convolve with PSF to smooth the curve and identify where peaks may be
    convolved = gaussian_filter1d(intensity, sigma)
    # Find Peaks in Convolved Image
    peaks, _ = find_peaks(convolved, distance=(sigma*2.355), height=40)

    # Deconvolve Image for Sharper Lines
    psf = GaussianPSF(255,3.5,[30, image.shape[1]])
    deconv_image = restoration.richardson_lucy(image[(row-15):(row+15), 0:image.shape[1]], psf, iterations=30)
    deconv = np.sum(deconv_image, 0)/30                                                                      # Create 1D Array using average intensity +-30 from center

    # Find x coords of max intensity in deconvolved image
    # Searches +-sigma from convolved peaks
    x_coords = []
    s = round(sigma)
    for peak in peaks:
        mx = np.where(deconv[(peak-s):(peak+s)] == np.max(deconv[(peak-s):(peak+s)]))[0][0] + (peak-s)
        x_coords.append(mx)

    # Create Array of Wavelengths
    wavelengths = []
    if isLinear:
        for val in x_coords:
            wavelengths.append(val*linearEq[0]+linearEq[1])
    else:
        for val in x_coords:
            wavelengths.append(pow(val,3)*cubicEq[0] + pow(val,2)*cubicEq[1] + val*cubicEq[2] + cubicEq[3])


    # Create Intensity Array
    intensity = []
    for i in peaks:
        intensity.append(convolved[i])
    return x_coords[:50], wavelengths[:50], intensity[:50]


#############################################################
#
# NonLinearCalibration
# - Analyzes fluorescent spectrum for cubic mapping/fit function
# - Returns cubicEq --> array
#
#############################################################

def nonLinearCalibration():
    # set the to desired florescent wavelength values to observe
    peak_values = [404.656, 435.833, 546.074, 611.878]
    x_coords, wavelengths, intensity = analyzeSpectrum(spectrum, avg_Y, True)

    # get postion of the wavelengths
    wavelengths_index = []

    wavelengths_index.append(closest(wavelengths, peak_values[0]))
    wavelengths_index.append(closest(wavelengths, peak_values[1]))
    wavelengths_index.append(closest(wavelengths, peak_values[2]))
    wavelengths_index.append(closest(wavelengths, peak_values[3]))
    
    print("Postion of the closest wavelength values")
    print(wavelengths_index)

    new_xCoords = []

    for i in wavelengths_index:
        new_xCoords.append(x_coords[i])
    
    print("Position of the updated pixel coordinates (x-values)")
    print(new_xCoords)

    nonLinearEq = np.polyfit(new_xCoords, peak_values, 3)

    print("Nonlinear Equation Values")
    print(nonLinearEq)
    
    return nonLinearEq
