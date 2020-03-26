import cv2
from tests.helper_functions import cropImage

testImagePath = './images/dng/signal/tiff-conv/LD011862_sig.tif'



croppedTestImage = cv2.imread(testImagePath)
testImage = cv2.imread(testImagePath)





croppedTestImage = cropImage(croppedTestImage)


oldY = croppedTestImage.shape[0]

oldX = croppedTestImage.shape[1]



enteredY = 10

newY = oldY - enteredY





cv2.circle(croppedTestImage, (oldX//2, oldY//2), 2, (255, 255, 0), 3)
cv2.imshow("pre", croppedTestImage)


cv2.waitKey(0)
cv2.destroyAllWindows()