import numpy as np
import matplotlib.pyplot as plt
import cv2                             # openCV used for audio and image processing
import sklearn                         #scikitlearn
# import tensorflow as tf

img = plt.imread("./photo1.jpg")

model = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")


# It is detecting faces on the images
all_faces = model.detectMultiScale(img)

for f in all_faces:
    x,y,w,h = f
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    # break

print(all_faces)
plt.imshow(img)
plt.show()