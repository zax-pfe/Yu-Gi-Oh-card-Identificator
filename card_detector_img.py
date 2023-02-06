import cv2
import numpy as np
from detector_functions import Card_detection
import os
from predict_functions import Predict
from keras import backend as K
from paddleocr import PaddleOCR # main OCR dependencies
from set_code_detector import Detect_setcode




ocr_model = PaddleOCR(lang='en')


threshold = 0.5
interval = 0.2
# model_trained_path = 'archive\\triplet_model_b_and_w_img244_epoch_5.h5'
# dict_trained_path = 'archive\\dict_black_white_244_padded (2).pkl'

model_trained_path = 'archive\\training_with_noise_28k.h5'
dict_trained_path = 'archive\\dict_28k.pkl'


setcode_detector = Detect_setcode(ocr_model)

predictor = Predict(model_trained_path,dict_trained_path,threshold,interval)

card_detector = Card_detection()


#pour chaque image dans le fichier image
for file in os.listdir('images'):
    path_image = "images\\" +file

    try:
        scanned_card, success = card_detector.return_scaned_card(path_image)
        if success==True:
            card_name = predictor.predict_card_name(scanned_card)

            try:
                setcode = setcode_detector.read_setcode(scanned_card)
            except:
                setcode = 'setcode-error'

            print("setcode : ", setcode)

            # print('setcode, card_name :',setcode, card_name[0] )

            card_detector.card_prediction(card_name, setcode) 
        else:
            print("error during detection")
            card_detector.error_detection(path_image, error_type=0)
    except:
        print("error")
        card_detector.error_detection(path_image, error_type=1)


