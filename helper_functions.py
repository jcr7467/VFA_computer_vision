import numpy as np
import cv2

class SortedCircle:
    def __init__(self, circleIn):
        self.circle = circleIn
        self.xCoord = circleIn[0]
        self.yCoord = circleIn[1]
        self.row = 0
        self.col = 0
        self.radius = circleIn[2]
        self.num_label = 0
        self.dist_from_origin = (circleIn[0]**2 + circleIn[1]**2) ** 0.5




def binarizeErodeAndDilate(croppedTestImage):
    ## IMAGEBLUR -> BINARIZE LATER
    #########################################################
    # Grayscale and binarization of testImage
    #########################################################
    # Might be really important ->  grayTestImage2 = testImage[:, :, 1]
    grayTestImage = cv2.cvtColor(croppedTestImage, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(grayTestImage, 13)

    # This is the binarized image
    bin_image2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 2)

    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(bin_image2, kernel, iterations=3)
    final_image = cv2.dilate(img_erosion, kernel, iterations=1)
    cv2.imshow("being binarized", final_image)
    bin_image2 = cv2.medianBlur(bin_image2, 3)
    return final_image


def cropImage(cropMe):
    print('its working!')
    return cropMe[1990:3400, 760:2430]


def houghTransform(manipulateMe, drawOnMe, drawBool):

    #print("This is the version: " + cv2.__version__)


    circles = cv2.HoughCircles(manipulateMe, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=80, maxRadius=110)
    circles = np.uint16(np.around(circles))

    listOfCoordinates = []

    for i in circles[0, :]:
        thick = 2

        listOfCoordinates.append(i)
        if drawBool:
            pass
            # This circles the outter circle, with a color of (0, 255,0) and a thickness of 2
            cv2.circle(drawOnMe, (i[0], i[1]), i[2], (0, 255, 0), thick)

            # This circles the center of the circle
            cv2.circle(drawOnMe, (i[0], i[1]), 2, (0, 255, 0), 3)
    return listOfCoordinates



def rotateAndScale(img, scaleFactor = 1, degreesCCW = 30):
    (oldY,oldX) = (img.shape[0], img.shape[1]) #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg


def rotateAndScale2(img, scaleFactor = 1, degreesCCW = 30):
    (oldY,oldX) = (img.shape[0], img.shape[1]) #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(183,215), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg


