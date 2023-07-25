import os
import cv2
from PIL import Image
import numpy as np
# 

# 
data=[]
labels=[]
# 
# ----------------
# LABELS
# Benign 0
# Malignant 1
# Normal 2
# 
# ----------------

# Benign 0
benign = os.listdir(os.getcwd() + "/CNN/data/benign")
for x in benign:
    imag=cv2.imread(os.getcwd() + "/CNN/data/benign/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

# Malignant 1
malignant = os.listdir(os.getcwd() + "/CNN/data/malignant/")
for x in malignant:
    imag=cv2.imread(os.getcwd() + "/CNN/data/malignant/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

# Normal 2
normal = os.listdir(os.getcwd() + "/CNN/data/normal/")
for x in normal:
    imag=cv2.imread(os.getcwd() + "/CNN/data/normal/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)



tumour=np.array(data)
labels=np.array(labels)
# 
np.save("tumour",tumour)
np.save("labels",labels)