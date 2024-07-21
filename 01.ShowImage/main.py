import cv2

img=cv2.imread('img.png')
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()