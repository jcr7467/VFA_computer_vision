import cv2
import numpy as np
import math





class Circle:
    def __init__(self, coord):
        self.coordinates = coord
        self.xCoord = coord[0]
        self.yCoord = coord[1]
        self.radius = coord[2]

def createCircles(listOfCoordinates):
    '''

    :param listOfCoordinates: The list created during hough transformation that gives us the list of circles
    :return:
    '''
    circleList = list()

    for coordinates in listOfCoordinates:
        newCircle = Circle(coordinates)
        circleList.append(newCircle)

    return circleList

def cropImage(cropMe):
    '''

    :param cropMe: image to be cropped
    :return:
     params of return statement are in [Y, X] cropping ranges
    '''

    return cropMe[1800:3400, 760:2430]

def houghTransformReturnCoords(manipulateMe, drawOnMe, drawBool):
    '''

    :param manipulateMe: the image that we are using to detect the circles.
    This is the image that has been binarized.
    :param drawOnMe: This is typically the original image. This is the image
     that will have the circles drawn on them, should the user elect to enable drawing
    :param drawBool: Determines if the circles are actually drawn. If not on,
     then we still have the coordinates of the circles it found.
    :return:
    '''

    circles = cv2.HoughCircles(manipulateMe, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=50, maxRadius=100)
    circles = np.uint16(np.around(circles))

    listOfCoordinates = []


    for i in circles[0, :]:
        lineThickness = 2
        colorOfCircles = (0, 255, 0)
        radiusForDot = 2
        radiusForCircle = i[2]
        xCoor = i[0]
        yCoor = i[1]

        listOfCoordinates.append(i)
        if drawBool:
            pass
            # This circles the outter circle, with a color of (0, 255,0) and a thickness of 2
            cv2.circle(drawOnMe, (xCoor, yCoor), radiusForCircle, colorOfCircles, lineThickness)

            # This circles the center of the circle
            cv2.circle(drawOnMe, (xCoor, yCoor), radiusForDot, colorOfCircles, lineThickness)
    return listOfCoordinates

def binarizeErodeAndDilate(transformMe):
    '''

    :param transformMe: this is the image that we will binarize and will return in the end
    :return:
    '''


    grayedImage = cv2.cvtColor(transformMe, cv2.COLOR_BGR2GRAY)
    blurredGrayImage = cv2.medianBlur(grayedImage, 13)
    #cv2.imshow("Gray", grayedImage)
    #cv2.imshow("Grayblur", blurredGrayImage)


    binImage = cv2.adaptiveThreshold(blurredGrayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 2)


    #cv2.imshow("Grayblur", binImage)



    return binImage

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

def shiftBy(deltaX, deltaY, img):

    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32([ [1,0,deltaX], [0,1,deltaY] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    return img_translation



def alignImage(coords, image):

    #This section finds the circles with the at the top of the image (the two alignment markers)
    circlesList = createCircles(coords)
    circlesList.sort(key=lambda Circle: Circle.yCoord)
    bottom1 = circlesList[-1].yCoord
    bottom2 = circlesList[-2].yCoord
    print(bottom1)
    print(bottom2)
    top1 = circlesList[0]
    top2 = circlesList[1]
    print(top1.yCoord)
    print(top2.yCoord)
    print(image.shape[0])

    if top1.xCoord > top2.xCoord:
        right1 = top1
        left1 = top2
    else:
        right1 = top2
        left1 = top1


    #This determines which way we need to rotate the image
    exp1 = int(right1.yCoord) - int(left1.yCoord)
    if exp1 > 0:
        direction = "CCW"
    else:
        direction = "CW"


    deltaY = abs(exp1)
    deltaX = right1.xCoord - left1.xCoord

    #print(deltaX, deltaY)
    #print(image.shape[1]) # 1670
    #print(image.shape[1] - deltaX)
    #print(586//2)



    phi = math.atan(deltaY/deltaX)
    if direction == "CW":
        phi = phi * -1


    #print("Radians: " + str(phi))
    #print("Degrees: " + str(phi * 180/math.pi))

    phi = phi * 180/math.pi
    rotatedImage = rotateAndScale(image, 1, phi)
    #cv2.imshow("Rotated", rotatedImage)
    print(293 - left1.xCoord)
    print(1377 - right1.xCoord)


    rotatedAndShiftedImage = shiftBy(293-left1.xCoord, 289 - left1.yCoord, rotatedImage)

    return rotatedAndShiftedImage

    '''
    print(left1.xCoord)
    print(right1.xCoord)
    shift = 293 - left1.xCoord

    rotateShift = shiftBy(shift, 0, rotate)
    cv2.imshow("RotatedShift", rotateShift)
    '''


def matchTemplate(image, template):


    template_dictionary = {
        'template_A': 'top_left_alignment_marker.tif',
        'template_B': 'top_right_alignment_marker.tif',
        'template_C': 'bottom_left_alignment_marker.tif',
        'template_D': 'bottom_right_alignment_marker.tif'
    }

    template = cv2.imread('fluorescent_templates/' + template_dictionary[template], cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w,h = template.shape[::-1]

    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)


    deltaX = bottom_right[0] - top_left[0]
    deltaY = bottom_right[1] - top_left[1]
    midXPoint = top_left[0] + deltaX//2
    midYPoint = top_left[1] + deltaY//2


    #cv2.rectangle(image, top_left, bottom_right, 255, 2)
    #cv2.circle(image, (midXPoint, midYPoint), 80, (255, 255, 0), 2)
    #cv2.circle(image, (midXPoint, midYPoint), 2, (255, 255, 0), 2)


    return (midXPoint, midYPoint)

def findAngle(alignA, alignB):

    deltaX = abs(int(alignA[0]) - int(alignB[0]))
    deltaY = int(alignA[1]) - int(alignB[1])

    if (deltaY) >= 0:
        direction = "CW"
    else:
        direction = "CCW"

    deltaY = abs(deltaY)

    return (math.atan(deltaY/deltaX), direction)

def rotateImage(image, alignA, alignB):

    angleToRotate, direction = findAngle(alignA, alignB)

    if direction == "CW":
        angleToRotate = -1 * angleToRotate * (180/math.pi)
    else:
        angleToRotate = angleToRotate * (180/math.pi)

    return rotateAndScale(image, 1, angleToRotate)

def alignImage2(image):


    alignA = matchTemplate(image, "template_A")
    alignB = matchTemplate(image, "template_B")
    rotated_image = rotateImage(image, alignA, alignB)

    cv2.imshow("h", rotated_image)
    new_alignA = matchTemplate(rotated_image, "template_A")

    cv2.circle(rotated_image, new_alignA, 80, (255, 255, 0), 2)

    cv2.imshow("h2", rotated_image)








    alignAX = new_alignA[0]
    alignAY = new_alignA[1]

    shifted = shiftBy(296 - alignAX, 291 - alignAY, rotated_image)
    cv2.circle(shifted, (296, 1308), 80, (255, 255, 0), 2)
    return shifted





def main():



    imagePath = 'fluorescent/image_6.tif'
    image = cv2.imread(imagePath)
    image = cropImage(image)
    cv2.imshow("Original", image)








    shifted = alignImage2(image)

    cv2.imshow("shifted", shifted)








    '''
    binarizedImage = binarizeErodeAndDilate(image)
    #cv2.imshow("Binarized", binarizedImage)

    coords = houghTransformReturnCoords(binarizedImage, image, True)
    cv2.imshow("Houghed", image)

    rotatedAndShifted = alignImage(coords, image)
    #cv2.imshow("RotatedShift", rotatedAndShifted)

    #val = matchTemplate(rotatedAndShifted)


    secBinarizedImage = binarizeErodeAndDilate(rotatedAndShifted)

    secCoords = houghTransformReturnCoords(secBinarizedImage, rotatedAndShifted, False)


    print(secCoords)
    cv2.circle(rotatedAndShifted, (289, 297), 80, (255, 255, 0), 2)
    cv2.circle(rotatedAndShifted, (1379, 297), 80, (255, 255, 0), 2)
    #cv2.imshow("Second Hough", rotatedAndShifted)


    '''


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()