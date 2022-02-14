from cgitb import grey
from re import I
from webbrowser import get
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context



X,y = fetch_openml('mnist_784',version = 1,return_X_y = True)
print(pd.Series(y).value_counts())
classes  = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

samples_per_classes = 5

figure = plt.figure(figsize = (nclasses*2,(1+samples_per_classes*2)))
idx_cls = 0

for cls in classes:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs,samples_per_classes,replace = False)
  i = 0
  for idx in idxs:
    plt_idx = i*nclasses+idx_cls+1
    p = plt.subplot(samples_per_classes,nclasses,plt_idx)
    p = sns.heatmap(np.array(X.loc[idx]).reshape(28,28),cmap = plt.cm.gray,
                    xticklabels = False, yticklabels = False, cbar = False)
    p = plt.axis('off')
    i += 1
  idx_cls += 1


x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
x_train_scaled = x_train/255.0

x_test_scaled = x_test/255.0

clf = LogisticRegression(solver= 'saga',multi_class = 'multinomial').fit(x_train_scaled,y_train)


y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2+56))
        cv2.rectange(gray,upper_left,bottom_right,(0,255,0),2)
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0],bottom_right[0]]

        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) 
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255) 
        max_pixel = np.max(image_bw_resized_inverted) 
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784) 
        test_pred = clf.predict(test_sample)
        
        print("Predicted class is: ", test_pred)

        cv2.imshow('frame',gray) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
    except Exception as e: 
        pass 

cap.release()
cv2.destroyAllWindows()








