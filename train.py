# YGO card Verification Using One Shot Learning

#### Importing Libraries

import tensorflow as tf 
import numpy as np
import os
from keras.layers import Conv2D, Activation, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Input, concatenate
import cv2
from keras import backend as K
from keras.layers import *
from keras.models import Model
K.set_image_data_format('channels_first')
import random
import matplotlib.pyplot as plt
import keras
import sys
import glob
import imgaug.augmenters as iaa
import glob

import pickle
import train_functions
import transform_image
import predict_functions
import model

"""#### Plot the versions"""

print("Keras version :",keras.__version__)
print("Tensorflow version :",tf.__version__)
print("Python version :",sys.version)

threshold = 0.5
dist = 0.1

"""#### Set if the model is already trained or not 

##### If you want to **train** the model, set **model_trained to False**
##### If you want to **test** the model, set **model_trained to True**

"""

model_trained = False

#If the model is already trained then define the path for the trained weight and the trained dict

#The second links is from the model trained with the negatives pictures no transformed
model_trained_path = 'archive\\triplet_model_b_and_w_img244_epoch_12.h5'
dict_trained_path = 'archive\dict_black_white_244_padded.pkl'

"""#### Defining the path of dataset"""

PATH='cardDatabaseFull'
images_path = glob.glob("cardDatabaseFull/*/*.jpg")

print("the size of the dataset is :",len(images_path))

model=model.FinalModel(input_shape=(3,244,244))

"""## Defining model for triplet loss"""

triplet_model_a=Input((3,244,244))
triplet_model_n=Input((3,244,244))
triplet_model_p=Input((3,244,244))
triplet_model_out=Concatenate()([model(triplet_model_a),model(triplet_model_p),model(triplet_model_n)])
triplet_model=Model([triplet_model_a,triplet_model_p,triplet_model_n],triplet_model_out)

triplet_model.compile(optimizer='adam',loss=train_functions.triplet_loss_t)

if model_trained == False:
    triplet_model.fit(train_functions.data_gen(),steps_per_epoch=330,epochs=15)
    triplet_model.save('triplet_model_b_and_w_img244_epoch_12.h5')

if model_trained == True:
    triplet_model=keras.models.load_model(model_trained_path,custom_objects={'triplet_loss_t':train_functions.triplet_loss_t})



"""### For images of same card"""

path_image = 'cardDatabaseFull\ABCDragon-Buster-0-1561110\\15611100.jpg'
path_rimage= 'image_croped_2\\1372666495.jpg_croped.jpg'

predict_functions.compare(path_rimage,triplet_model = triplet_model, path_image2=path_image)


"""### For images of different cards"""

path_rimage = 'image_croped_2\\1372666495.jpg_croped.jpg'
path_image= 'cardDatabaseFull\8Claws-Scorpion-0-14261867\142618670.jpg'

predict_functions.compare(path_rimage,triplet_model = triplet_model, path_image2=path_image)


"""# Creation of a dictionary containing the name of the card and the output of the model when we passed in the given card

#### Get the name of the folder to have the name of the card
"""

# this function take the images path and return only the name of the card

def return_card_name(img_path):
    card_folder = img_path.split('/')[5] # the name of the card is at the 6th position
    card_name = card_folder.split('-0-')[0]
    card_name = card_name.replace('-', ' ')
    return card_name

"""#### Creation of a list of tuples : (card name , tensor) in order to create then the dictionary"""

if model_trained == False:
    
    tuples_names_emb_list = []
    for i in (images_path):
        img=transform_image.image_resizing(i)
        tuples_names_emb_list.append((return_card_name(i), predict_functions.encode_img(img,triplet_model)))
    names_and_emb_dict = {key: value for (key, value) in tuples_names_emb_list}

"""#### Now we save the dictionnay"""

if model_trained == False:
    
    # Open a file for writing
    with open('dict_black_white_244_padded.pkl', 'wb') as f:
      # Write the dictionary to the file
      pickle.dump(names_and_emb_dict, f)

"""#### If the model is already trained and the dict already created, we just have to load it"""

if model_trained == True:
    
    with open(dict_trained_path, 'rb') as f:
      # Load the dictionary from the file
      names_and_emb_dict = pickle.load(f)

    # Print one element of the dictionary to test it
#     print(names_and_emb_dict['Gagaga Head'])

"""# Compare the output of the model with the dictionnary"""



"""#### Test of the model """

card_path_to_test = 'image_croped_2\WhatsApp Image 2023-01-13 at 10.19.23 (1).jpeg_croped.jpg'

card_name = predict_functions.compare_img_with_dict(card_path_to_test, names_and_emb_dict, triplet_model)
print(card_name)

# print time
# print(%timeit)

