import cv2
import numpy as np
import pdb
contours = np.array([[50,50], [50,150], [150,150], [150,50]])
image = np.zeros((200,200))
cv2.fillPoly(image, pts = [contours], color =(255,255,255))
cv2.imshow("filledPolygon", image)

