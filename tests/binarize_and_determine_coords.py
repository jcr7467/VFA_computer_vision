import cv2
import numpy as np


from tests.crop_images import cropImage2





def findCoordinates():

    testImagePath = './images/dng/signal/tiff-conv/LD011839_sig.tif'
    croppedTestImage = cv2.imread(testImagePath)
    croppedTestImage = cropImage2(croppedTestImage)


    grayTestImage2 = cv2.cvtColor(croppedTestImage, cv2.COLOR_BGR2GRAY)
    img2 = cv2.medianBlur(grayTestImage2, 13)

    bin_image2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 2)



    kernel = np.ones((2,2), np.uint8)
    img_erosion = cv2.erode(bin_image2, kernel, iterations=3)
    final_image = cv2.dilate(img_erosion, kernel, iterations=1)

    toBetested = final_image

    circles = cv2.HoughCircles(toBetested, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=80, maxRadius=110)
    circles = np.uint16(np.around(circles))

    listOfCoordinates = []



    for i in circles[0, :]:
        thick = 2

        listOfCoordinates.append(i)
        # This circles the outter circle, with a color of (0, 255,0) and a thickness of 2
        cv2.circle(croppedTestImage, (i[0], i[1]), i[2], (0, 255, 0), thick)

        # This circles the center of the circle
        cv2.circle(croppedTestImage, (i[0], i[1]), 2, (0, 255, 0), 3)


    cv2.imshow("Final image", croppedTestImage)



    return listOfCoordinates