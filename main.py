# https://thecleverprogrammer.com/2020/10/09/face-detection-with-python/
'''
https://pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/
https://www.thepythoncode.com/article/predict-age-using-opencv
'''
import cv2

face_cascade = cv2.CascadeClassifier('frontalface.xml')
img = cv2.imread('people.jpg')
faces = face_cascade.detectMultiScale(img, 1.1, 4)

for (x, y, w, h) in faces: 
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite("people_detected.png", img) 
print('Successfully saved')
