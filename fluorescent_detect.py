import cv2
import numpy as np
import math






def cropImage(cropMe):
    '''

    :param cropMe: image to be cropped
    :return:
     params of return statement are in [Y, X] cropping ranges
    '''

    return cropMe[1800:3400, 760:2430]



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

    #cv2.imshow("h", rotated_image)
    new_alignA = matchTemplate(rotated_image, "template_A")

    cv2.circle(rotated_image, new_alignA, 80, (255, 255, 0), 2)




    alignAX = new_alignA[0]
    alignAY = new_alignA[1]

    shifted = shiftBy(296 - alignAX, 291 - alignAY, rotated_image)
    #cv2.circle(shifted, (296, 1308), 80, (255, 255, 0), 2)
    return shifted





def main():



    imagePath = 'fluorescent/image_6.tif'
    image = cv2.imread(imagePath)
    image = cropImage(image)
    #cv2.imshow("Original", image)








    shifted = alignImage2(image)




    cv2.circle(shifted, (690, 440), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (985, 440), 60, (255, 155, 70), 2)



    cv2.circle(shifted, (445, 665), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (690, 665), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (985, 665), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (1225, 665), 60, (255, 155, 70), 2)



    cv2.circle(shifted, (835, 800), 60, (255, 155, 70), 2)



    cv2.circle(shifted, (445, 935), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (690, 935), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (985, 935), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (1225, 935), 60, (255, 155, 70), 2)

    cv2.circle(shifted, (690, 1170), 60, (255, 155, 70), 2)
    cv2.circle(shifted, (985, 1170), 60, (255, 155, 70), 2)

    cv2.imshow("shifted", shifted)


    '''Convert image to grayscale using cvtColor. You will get a matrix of values between 0 to 255. Get the average over 
    the values of this matrix. '''




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