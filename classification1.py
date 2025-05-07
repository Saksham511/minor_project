import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
from keras.preprocessing import image
import keras

from keras.models import load_model
model = load_model("best_model.h5")

def classify():
    #Giving path for image
    path= "savedImage.jpg"
    img = load_img(path, target_size=(256,256))

    i = img_to_array(img)
    i = preprocess_input(i)

    input_arr = np.array([i])
    input_arr.shape

    pred = np.argmax(model.predict(input_arr))

    if pred == 0:
        return 0
    elif pred == 1:
        return 1
    else:
        return 2

    #DISPLAYING IMAGE AND CLASSIFYING  