import cv2
import numpy as np
import dlib

'''img = cv2.imread('data/images/01.jpg')
img = cv2.resize(img,(0,0),None,0.5,0.5)
imgOriginal = img.copy()'''

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/models/shape_predictor_68_face_landmarks.dat')
def empty(a):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 640, 240)
cv2.createTrackbar("Blue", "BGR", 0, 255, empty)
cv2.createTrackbar("Green", "BGR", 0, 255, empty)
cv2.createTrackbar("Red", "BGR", 0, 255, empty)

def createBox(img,points,scale=5,masked=False,cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        img = cv2.bitwise_and(img,mask)
    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        return imgCrop
    else:
        return mask

while True:
    img = cv2.imread('data/images/01.jpg')
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        #imgOriginal = cv2.rectangle(img, (x1, y1), (x2, y2),(0, 255, 0), cv2.FILLED)
        landmarks = predictor(imgGray, face)
        mypoints = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mypoints.append([x, y])
            #cv2.circle(imgOriginal, (x, y), 3, (50, 50, 255), -1)
            #cv2.putText(imgOriginal, str(n), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)
        mypoints = np.array(mypoints)
        #imgLeftEye = createBox(img, mypoints[36:42])
        imgLips = createBox(img, mypoints[48:61],3,masked=True,cropped=False)

        imgColorLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos("Blue", "BGR")
        g = cv2.getTrackbarPos("Green", "BGR")
        r = cv2.getTrackbarPos("Red", "BGR")
        imgColorLips[:] = b, g, r  #BGR
        imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
        imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
        #imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2GRAY)
        imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)


        cv2.imshow('BGR', imgColorLips)
        #cv2.imshow('LeftEye', imgLeftEye)
        cv2.imshow('Lips', imgLips)


        print(mypoints)


    cv2.imshow('Original', imgOriginal)
    cv2.waitKey(0)