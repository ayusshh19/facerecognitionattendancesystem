import cv2
import face_recognition
import numpy as np
from django.shortcuts import render

imgjanhavi = face_recognition.load_image_file('j.photo.jpeg')
imgjanhavi = cv2.cvtColor(imgjanhavi,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('janh.png')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)


faceloc = face_recognition.face_locations(imgjanhavi)[0]
encodejanhavi = face_recognition.face_encodings(imgjanhavi)[0]
cv2.rectangle(imgjanhavi,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facetest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(facetest[3],facetest[0]),(facetest[1],facetest[2]),(255,0,255),2)

result =face_recognition.compare_faces([encodejanhavi],encodetest)

cv2.imshow('Janhavi',imgjanhavi)
cv2.imshow('Test',imgtest)
cv2.waitKey(0)