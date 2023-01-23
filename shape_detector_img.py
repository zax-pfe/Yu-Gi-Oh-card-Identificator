import cv2
import numpy as np
from functions import Card_detection
import os
from predict_functions import Predict
from keras import backend as K

threshold = 0.5
interval = 0.2
model_trained_path = 'archive\\triplet_model_b_and_w_img244_epoch_5.h5'
dict_trained_path = 'archive\\dict_black_white_244_padded (2).pkl'

predictor = Predict(model_trained_path,dict_trained_path,threshold,interval)

card_detector = Card_detection()


#pour chaque image dans le fichier image
for file in os.listdir('images'):
    path_image = "images\\" +file
    scanned_card = card_detector.return_scaned_card(path_image)
    card_name = predictor.predict_card_name(scanned_card)
    card_detector.card_prediction(card_name) 

