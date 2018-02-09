#python
# coding: utf-8

import cv2
from PIL import Image

import numpy as np
import os, time, glob


# In[23]:

print("Starting")
#Directory locations for the code
image_path = "Data\\img_align_celeba"

#Cropped Output
output_path = "Data\\cropped_faces"
if not os.path.exists(output_path):
        os.mkdir(output_path)
        

##### Cropping just the face of the celebrities #####
## Reference: stackoverflow ##

###Takes 1 hr to run for 200k Images ####

##Requires haarcascade_frontalface_default.xml file ##
face_cascade = "Data\\xml\\haarcascade_frontalface_default.xml"

def crop_and_save_images(cascade, imgname, output_path):
    img = cv2.imread(os.path.join(image_path, imgname))
    for i, face in enumerate(cascade.detectMultiScale(img)):
        #start = time.time()
        x, y, w, h = face
        sub_face = img[y:y + h, x:x + w]
        img_path = os.path.join(output_path, imgname)
        cv2.imwrite(img_path, sub_face)
        #if i%1000 == 0:
            #end = time.time() - start
            #print(i,"took", end, "seconds")
            #print(i, "Cropped")
            #start = time.time()
		 
#Cropping the images to contain only faces
cascade = cv2.CascadeClassifier(face_cascade)
count = 0
for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
    crop_and_save_images(cascade, f, output_path)
    if(count%1000 == 0):
        print(count)
    count += 1