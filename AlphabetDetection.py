import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X, y = fetch_openml('mnist_784',version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver = "saga",
multi_class = "multinomial").fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(acc*100)
cap = cv2.VideoCapture(0)

while True:
  try:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height,width = gray.shape
    upper_left = (int(width/2-56), int(height/2-56))
    bottom_right = (int(width/2+56), int(height/2+56))
    cv2.rectangle(gray, upper_left, bottom_right, (0,255,0),2)
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]


    im_PIL = Image.fromArray(roi)
    image = im_PIL.convert("L")
    img_resized = image.rezize(28, 28)
    img_resize_invert = PIL.ImageOps.invert(img_resized)
    pixel_filter = 20
    min_pxl = np.percentile(img_resize_invert, pixel_filter)
    img_resize_invert_scale = np.clip(img_resize_invert-min_pxl, 0 , 255)
    max_pxl = np.max(img_resize_invert)

    
    img_resize_invert_scale = np.asarray(img_resize_invert_scale)/max_pxl
    test_sample = np.array(img_resize_invert_scale).reshape(1, 784)
    test_pred = clf.predict(test_sample)
    print(test_pred)
    cv2.imshow("frame", gray)
    cv2.waitKey(0)
  except Exception as e:
    pass
cap.release()
cv2.destroyAllWindows()  