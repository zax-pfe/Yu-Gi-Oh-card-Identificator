import cv2
import numpy as np
import functions
import os
from predict_functions import Predict
from keras import backend as K

threshold = 0.5
interval = 0.2
model_trained_path = 'archive\\triplet_model_b_and_w_img244_epoch_5.h5'
dict_trained_path = 'archive\\dict_black_white_244_padded (2).pkl'

predictor = Predict(model_trained_path,dict_trained_path,threshold,interval)



def plot_card_and_name(path_image):

    img = cv2.imread(path_image)
    resized,width, height = functions.return_resized_img_percent(img, 20)
    card_plus_name, _, _=functions.return_resized_img_percent(img, 40)
    original = img.copy()


    height_original,width_original = original.shape[:2]
        
    """ Processing for card detection - resized """
    imgDil = functions.transform_for_contour_detection(resized)
    # this function get the contours and create a pproximative bounding box
    contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # this fucntion is used to scan the card 
    biggest, _ = functions.biggestContour(contours)


    if biggest.size !=0:
        biggest=functions.reorder(biggest)

        #get the size of the img and the 4 points at the corner to prepare the scan
        points = np.int32(biggest)
        pts1 = np.float32(biggest)
        pts1_original = pts1*5
        
        pts2 = np.float32([[0, 0],[width, 0], [0, height],[width, height]])
        pts2_original = pts2 * 5
 
        # get a list of the 4 points a the corner 
        point_list = functions.extract_points(points)        

        matrix_original = cv2.getPerspectiveTransform(pts1_original, pts2_original)              
        imgWarpColored_original = cv2.warpPerspective(original, matrix_original, (width_original, height_original))

    card_name = predictor.predict_card_name(imgWarpColored_original)
    print(card_name[0][0])

    point_for_card_plus_name = [[x * 2 for x in sublist] for sublist in point_list]
    card_plus_name = functions.add_card_name(card_plus_name, card_name[0][0],point_for_card_plus_name)
    cv2.imshow('card_plus_name', card_plus_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#pour chaque image dans le fichier image
for file in os.listdir('images'):
    path_image = "images\\" +file
    plot_card_and_name(path_image)
 